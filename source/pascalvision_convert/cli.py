# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Command-line entry point for pascalvision-convert."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

from .convert import Target, convert_onnx, convert_tflite
from . import pipeline, recipe
from .tuning import DEFAULT_TRIALS_PER_TASK


def _parse_shapes(items: List[str]) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    for item in items:
        
        
        
        
        
        
        
        name, _, dims = item.rpartition(":")
        if not name or not dims:
            raise ValueError(f"bad --shape '{item}', expected NAME:D0,D1,...")
        shapes[name] = tuple(int(d) for d in dims.split(",") if d)
    return shapes


def main(argv: Optional[List[str]] = None) -> int:
    incoming = list(argv) if argv is not None else list(sys.argv[1:])
    if incoming and incoming[0] == 'ocr-composite':
        
        
        
        
        from . import ocr_composite
        try:
            return ocr_composite._main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'ocr-composite-audit':
        
        
        
        
        
        from . import ocr_composite_oracle
        try:
            return ocr_composite_oracle._main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'ocr-ort-raw':
        
        
        
        
        
        from . import ocr_ort_raw_capture
        try:
            return ocr_ort_raw_capture.main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'ocr-det-pairwise':
        
        
        
        
        from . import ocr_det_pairwise
        try:
            return ocr_det_pairwise.main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'ocr-png-input':
        
        
        
        
        from . import ocr_png_preprocess
        try:
            return ocr_png_preprocess.main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'destination-native-cpu':
        
        
        
        
        from . import destination_native_runner
        try:
            return destination_native_runner.main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'remote-cpu-contract':
        
        
        
        from . import remote_memtest
        try:
            return remote_memtest.main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'omdet-candidate-bundle':
        
        
        
        
        
        from . import omdet_public_candidate
        try:
            return omdet_public_candidate.main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'support-inventory':
        
        
        
        from . import support_inventory
        try:
            return support_inventory.main(incoming[1:])
        except SystemExit as exc:
            if type(exc.code) is int:
                return exc.code
            return 1
    if incoming and incoming[0] == 'birefnet-hr800-preview':
        
        
        from . import birefnet_hr800_route
        return birefnet_hr800_route.preview_main(incoming[1:])
    if incoming and incoming[0] == 'birefnet-hr800-held':
        
        
        
        
        from . import birefnet_hr800_route
        return birefnet_hr800_route.main(incoming[1:])
    if incoming and incoming[0] == 'omdet-held-preview':
        
        
        
        from . import omdet_compiler_route
        return omdet_compiler_route.preview_main(incoming[1:])
    if incoming and incoming[0] == 'omdet-held-compiler':
        
        
        
        from . import omdet_compiler_route
        return omdet_compiler_route.main(incoming[1:])
    if incoming and incoming[0] == 'hf-selected-preview':
        
        
        
        from . import hf_selected_cli
        return hf_selected_cli.preview_main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-preview':
        
        
        
        from . import paddle_selected_cli
        return paddle_selected_cli.preview_main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-rec-input':
        
        
        
        
        from . import paddle_selected_rec_input
        return paddle_selected_rec_input.main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-rec-original-ort':
        
        
        
        from . import paddle_selected_rec_original_ort
        return paddle_selected_rec_original_ort.main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-rec-pairwise':
        
        
        
        from . import paddle_selected_rec_pairwise
        return paddle_selected_rec_pairwise.main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-det-input':
        
        
        
        from . import paddle_selected_det_input
        return paddle_selected_det_input.main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-det-original-ort':
        
        
        
        from . import paddle_selected_det_original_ort
        return paddle_selected_det_original_ort.main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-det-pairwise':
        
        
        
        
        from . import paddle_selected_det_pairwise
        return paddle_selected_det_pairwise.main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-native-preview':
        
        
        from . import paddle_selected_native
        return paddle_selected_native.preview_main(incoming[1:])
    if incoming and incoming[0] == 'paddle-selected-native-run':
        
        from . import paddle_selected_native
        return paddle_selected_native.run_main(incoming[1:])
    if '--recipe' in incoming or '--into' in incoming:
        
        
        
        if '-o' in incoming or '--output' in incoming:
            print('error: choose either --into (complete bundle) or the '
                  'legacy -o form, not both', file=sys.stderr)
            return 1
        return recipe.main(incoming)

    parser = argparse.ArgumentParser(
        prog="pascalvision-convert",
        description="Build a complete TVM bundle using the shared recipe engine.",
    )
    parser.add_argument("model", help="path to the .onnx or .tflite model")
    parser.add_argument("-o", "--output", required=True, help="output module path (.so)")
    parser.add_argument(
        "--target",
        choices=[t.value for t in Target],
        default=Target.CPU.value,
        help="execution target (default: cpu)",
    )
    parser.add_argument("--opt-level", type=int, default=3, help="Relax optimization level 0-3")
    parser.add_argument('--sm', default='', help='requested CUDA SM, e.g. sm_89')
    parser.add_argument('--target-os', default='', help='deployment operating system')
    parser.add_argument('--target-cpu-arch', default='', help='deployment CPU architecture')
    parser.add_argument('--target-driver-api', default='', help='CUDA Driver API requirement')
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="NAME:D0,D1,...",
        help="override an input shape (repeatable)",
    )
    parser.add_argument(
        "--tune",
        metavar="WORK_DIR",
        default=None,
        help=("build with code tuning, keeping the tuning database in WORK_DIR. "
              "CPU only. Measured 01.09.2026: 32 to 58 times faster than an "
              "untuned build, and the budget is derived from the model's own "
              "task count - see --trials-per-task"),
    )
    parser.add_argument(
        "--trials-per-task",
        type=int,
        default=DEFAULT_TRIALS_PER_TASK,
        metavar="N",
        help=("trials each tuning task receives (default: %(default)s). The "
              "total budget is N times the number of tasks; a smaller figure "
              "leaves tasks untuned, which is slower than not tuning at all"),
    )
    args = parser.parse_args(incoming)

    try:
        shapes = _parse_shapes(args.shape)
        if args.target == 'cudnn':
            print('error: cudnn BYOC is a legacy low-level path; shared '
                  'BuildPlan cannot package/verify it yet. No CUDA fallback.',
                  file=sys.stderr)
            return 1
        if args.opt_level != 3:
            print('error: --opt-level other than 3 is not supported by the '
                  'complete bundle pipeline; no ignored optimization option',
                  file=sys.stderr)
            return 1
        if len(shapes) > 1:
            print('error: only one input shape is accepted by the current '
                  'bundle plan; do not silently ignore other tensor shapes',
                  file=sys.stderr)
            return 1
        if '--trials-per-task' in incoming and not args.tune:
            print('error: --trials-per-task has no effect without --tune',
                  file=sys.stderr)
            return 1
        if args.tune and args.target != 'cpu':
            print('error: --tune currently applies to cpu only',
                  file=sys.stderr)
            return 1
        if args.tune and args.model.lower().endswith('.tflite'):
            print('error: --tune has no TFLite implementation',
                  file=sys.stderr)
            return 1
        suffix = '.' + args.target + pipeline.module_extension()
        named = os.path.basename(args.output)
        if not named.endswith(suffix) or len(named) <= len(suffix):
            print('error: legacy -o must use a canonical native module name '
                  'such as model%s; --into is recommended for a full bundle'
                  % suffix, file=sys.stderr)
            return 1
        name = named[:-len(suffix)]
        directory = os.path.dirname(os.path.abspath(args.output))
        routed = [args.model, '--into', directory, '--target', args.target,
                  '--name', name]
        if args.sm:
            routed += ['--sm', args.sm]
        if args.target_os:
            routed += ['--target-os', args.target_os]
        if args.target_cpu_arch:
            routed += ['--target-cpu-arch', args.target_cpu_arch]
        if args.target_driver_api:
            routed += ['--target-driver-api', args.target_driver_api]
        if args.shape:
            routed += ['--shape', args.shape[0]]
        if args.tune:
            routed += ['--tune', args.tune,
                       '--trials-per-task', str(args.trials_per_task)]
        return recipe.main(routed)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_with_a_large_stack(argv=None) -> int:
    """Run main() on a thread with a big stack.

    TVM lowers a Relax module by recursive descent, and on a deep graph that
    recursion outgrows the default thread stack. The process then dies without an
    exception: on Windows the exit code is 0xC00000FD, STACK_OVERFLOW.

    Measured 31.08.2026 on BiRefNet (birefnet-lite, 16400 nodes, 4325 after
    constant folding). Converting it on the main thread died with exit code
    -1073741571 and no message; the last thing printed was an unrelated TOPI
    warning about fast-mode gather, which for months was mistaken for the cause.
    The same model on a 192 MB stack parses, runs the pipeline and builds.

    Windows refused 256 MB, accepted 192 MB, so that is the size asked for; if a
    platform refuses even that, the conversion still runs, just on the default
    stack, and a deep model will fail the way it used to rather than not at all.
    """
    import threading

    outcome = {}

    def worker() -> None:
        outcome["code"] = main(argv)

    for megabytes in (192, 128, 64):
        try:
            threading.stack_size(megabytes * 1024 * 1024)
            break
        except (ValueError, RuntimeError):
            continue
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return outcome.get("code", 1)


if __name__ == "__main__":
    raise SystemExit(_run_with_a_large_stack())
