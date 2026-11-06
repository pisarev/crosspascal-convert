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
                if command != 'generate':
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
            try:
                if message.get('images'):
                    raise ValueError('This Gemma 3n bundle supports text only; image input requires a vision converter')
                prompt = message.get('prompt')
                limit = message.get('max_new_tokens', 0)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError('Enter a non-empty prompt')
                if type(limit) is not int or limit < 0:
                    raise ValueError('Use 0 for automatic generation or a positive manual limit')
                folder = Path(message['bundle']).resolve(strict=True)
                manifest = json.loads((folder / 'model.json').read_text(encoding='utf-8'))
                if manifest.get('backend') != 'TVM CUDA W8A32':
                    raise ValueError('Select a Gemma 3n TVM CUDA W8A32 bundle')
                identity = (str(folder), (folder / 'model.json').read_bytes())
                reused = identity == loaded and session is not None
                if not reused:
                    emit('status', request_id, text='Loading the Gemma 3n CUDA model...')
                    if session is not None:
                        session.close()
                        session, loaded = None, None
                    session = GpuSession(folder)
                    loaded = identity
                emit('loaded', request_id, reused=reused, context=session.manifest['context'],
                     gpu=session.device.device_name, load_seconds=session.load_seconds)
                result = session.generate(prompt, limit,
                    on_text=lambda text: emit('text', request_id, text=text),
                    should_stop=stopping.is_set, chat=not message.get('plain', False))
                result['model_reused'] = reused
                result['modalities'] = ['text']
                emit('result', request_id, result=result)
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
