# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import argparse
import json
import os
import sys

from . import hf_closure


def preview_main(argv=None):
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert hf-selected-preview',
        description='Read-only selected Qwen/Gemma weights+tokenizer SHA; '
                    'no compiler, native model oracle, READY or release')
    parser.add_argument('--source-dir', required=True, metavar='ABS_HF_DIRECTORY')
    parser.add_argument('--model-key', required=True,
                        choices=('qwen2.5-0.5b', 'qwen2.5-1.5b',
                                 'qwen2-compatible-local', 'gemma-2-2b'))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        if not os.path.isabs(args.source_dir):
            raise ValueError('selected HF Preview requires an absolute local source')
        selected = hf_closure.scan_selected_hf_source(
            args.source_dir, expected_model_key=args.model_key)
        if (selected['model_key'] != args.model_key or
                selected['status'] != 'SCANNED_NOT_LOCKED' or
                selected['verification_status'] != 'NOT_CHECKED' or
                selected['release_grade_immutable'] is not False):
            raise ValueError('HF source scanner returned non-advisory selection')
        print(json.dumps(selected, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False))
        return 2
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        print('HF SELECTED PREVIEW REFUSED, NOT_CHECKED: %s' % exc,
              file=sys.stderr)
        return 1
