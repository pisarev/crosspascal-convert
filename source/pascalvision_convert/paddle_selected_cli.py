# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Preview the files and SHA-256 of a local Paddle inference model.

The scan does not hold source files open or validate model execution."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import paddle_closure


def preview_main(argv=None):
    parser = argparse.ArgumentParser(
        prog='pascalvision-convert paddle-selected-preview',
        description='Read-only exact selected Paddle inference graph/params/'
                    'optional metadata/dictionary SHA, not trained acceptance')
    parser.add_argument('--source-dir', required=True,
                        metavar='ABS_PADDLE_INFERENCE_DIRECTORY')
    parser.add_argument('--role', required=True, choices=('det', 'rec'))
    parser.add_argument('--dictionary', default='', metavar='ABS_REC_UTF8_DICTIONARY',
                        help='explicit required recognition charset; Det forbids one')
    parser.add_argument('--expected-dictionary-sha256', default='', metavar='SHA256',
                        help='optional independent SHA of caller-selected Rec charset')
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:

        return 0 if error.code == 0 else 1
    try:
        if not os.path.isabs(args.source_dir):
            raise ValueError('Paddle selected Preview requires absolute local source')
        chosen = paddle_closure.scan_selected_paddle_source(
            args.source_dir, role=args.role,
            dictionary=args.dictionary or None,
            expected_dictionary_sha256=args.expected_dictionary_sha256 or None)
        if (chosen['role'] != args.role or
                chosen['status'] != 'SCANNED_NOT_LOCKED' or
                chosen['verification_status'] != 'NOT_CHECKED' or
                chosen['release_grade_immutable'] is not False):
            raise ValueError('Paddle selection cannot assert compiler/native status')
        print(json.dumps(chosen, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False))
        return 2
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print('PADDLE SELECTED PREVIEW REFUSED / NOT_CHECKED: %s' % error,
              file=sys.stderr)
        return 1
