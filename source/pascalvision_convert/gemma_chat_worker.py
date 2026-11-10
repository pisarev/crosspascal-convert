# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
import json
import os
from pathlib import Path
import queue
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.pop('CROSSPASCAL_GEMMA_RUNTIME_PACKAGES', None)


def main():
    from pascalvision_convert.gemma3n_gpu_runtime import GpuSession
    from pascalvision_convert.gemma3n_runtime import Session as Cpu3nSession
    from pascalvision_convert.gemma2_runtime import Session as Cpu2Session
    output = threading.Lock()
    state_lock = threading.Lock()
    pending = queue.Queue()
    stopping = threading.Event()
    closing = threading.Event()
    current = [None]

    def emit(kind, request_id=None, **values):
        with output:
            print(json.dumps(dict(event=kind, id=request_id, **values), ensure_ascii=True), flush=True)

    def receive():
        try:
            while True:
                line = sys.stdin.buffer.readline(1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 1024 * 1024 or not line.endswith(b'\n'):
                    raise ValueError('Request exceeds the protocol limit')
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError('Expected a JSON request object')
                command = message.get('command')
                if command == 'close':
                    break
                if command == 'stop':
                    with state_lock:
                        if message.get('id') == current[0]:
                            stopping.set()
                    continue
                if command not in ('generate', 'load', 'unload'):
                    emit('error', message.get('id'), message='Unknown command')
                    continue
                request_id = message.get('id')
                if type(request_id) is not int or request_id < 1:
                    raise ValueError('Expected a positive request ID')
                with state_lock:
                    if current[0] is not None:
                        emit('error', request_id, message='Generation is already active')
                        continue
                    current[0] = request_id
                    stopping.clear()
                pending.put(message)
        except Exception as exc:
            emit('fatal', message=str(exc))
        finally:
            closing.set()
            stopping.set()
            pending.put(None)

    emit('ready', protocol='crosspascal.gemma-chat.v1', modalities=['text'])
    reader = threading.Thread(target=receive, name='GemmaCommands', daemon=True)
    reader.start()
    session, loaded = None, None
    try:
        while True:
            message = pending.get()
            if message is None or closing.is_set():
                break
            request_id = message['id']
            reused, images = False, []
            try:
                if message['command'] == 'unload':
                    if session is not None:
                        session.close()
                    session, loaded = None, None
                    emit('unloaded', request_id)
                    continue
                prompt = message.get('prompt')
                limit = message.get('max_new_tokens', 0)
                if message['command'] == 'generate' and (not isinstance(prompt, str) or not prompt.strip()):
                    raise ValueError('Enter a non-empty prompt')
                if type(limit) is not int or limit < 0:
                    raise ValueError('Use 0 for automatic generation or a positive manual limit')
                folder = Path(message['bundle']).resolve(strict=True)
                manifest = json.loads((folder / 'model.json').read_text(encoding='utf-8'))
                images = message.get('images', [])
                if not isinstance(images, list) or any(not isinstance(p, str) for p in images):
                    raise ValueError('Images must be a list of file paths')
                gemma_images = (manifest.get('architecture') == 'gemma3n' and
                                manifest.get('vision_manifest') == 'vision.json' and
                                manifest.get('backend') in ('TVM CUDA W8A32', 'TVM LLVM CPU FP32'))
                qwen_images = (manifest.get('architecture') == 'qwen2_5_vl' and
                               manifest.get('schema') == 'crosspascal.qwen2_5_vl.v1' and
                               manifest.get('backend') == 'TVM LLVM CPU FP32')
                if images and not (gemma_images or qwen_images):
                    raise ValueError('This bundle accepts text only. Remove the image or select a converted Qwen2.5-VL or Gemma 3n image bundle.')
                if qwen_images:
                    from pascalvision_convert.qwen_vl_runtime import Session as QwenVLSession
                    factory = QwenVLSession
                elif manifest.get('backend') == 'TVM CUDA W8A32':
                    factory = GpuSession
                elif manifest.get('architecture') == 'gemma3n':
                    factory = Cpu3nSession
                elif manifest.get('model_type') == 'gemma2':
                    factory = Cpu2Session
                else:
                    raise ValueError('Select a converted Qwen2.5-VL CPU, Gemma 2 CPU or Gemma 3n CPU/CUDA bundle')
                identity = (str(folder), (folder / 'model.json').read_bytes())
                reused = identity == loaded and session is not None
                if not reused:
                    emit('status', request_id, text='Loading the selected language model...')
                    if session is not None:
                        session.close()
                        session, loaded = None, None
                    session = factory(folder)
                    loaded = identity
                emit('loaded', request_id, reused=reused,
                     context=session.manifest.get('context', session.manifest.get('context_window_size')),
                     gpu=session.device.device_name if hasattr(session, 'device') else 'CPU',
                     load_seconds=getattr(session, 'load_seconds', None),
                     modalities=['text', 'image'] if getattr(session, 'vision', None) else ['text'])
                if message['command'] == 'load':
                    continue
                instructions = message.get('system_prompt', '')
                if not isinstance(instructions, str):
                    raise ValueError('System instructions must be text')
                if instructions:
                    prompt = instructions + '\n\n' + prompt
                extra = {'images': images} if images else {}
                result = session.generate(prompt, limit,
                    on_text=lambda text: emit('text', request_id, text=text),
                    should_stop=stopping.is_set, chat=not message.get('plain', False),
                    options=message.get('options'), **extra)
                result['model_reused'] = reused
                result['modalities'] = ['text', 'image'] if images else ['text']
                emit('result', request_id, result=result)
            except InterruptedError:
                emit('result', request_id, result={'text': '', 'generated_tokens': 0,
                     'stop': 'user_stop', 'model_reused': reused,
                     'modalities': ['text', 'image'] if images else ['text']})
            except Exception as exc:
                emit('error', request_id, message=f'{type(exc).__name__}: {exc}')
            finally:
                with state_lock:
                    current[0] = None
                emit('done', request_id)
    finally:
        if session is not None:
            session.close()
    emit('closed')


if __name__ == '__main__':
    main()
