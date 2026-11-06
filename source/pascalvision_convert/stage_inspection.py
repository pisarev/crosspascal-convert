# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Offline, read-only inspection of retained stages after a server restart.

This deliberately does NOT claim stage ownership, restore READY, authorize
deletion, recurse into child data, or execute/verify a native model. It only
reports candidate sibling directory names and their observed NTFS FileIDs.
No persistent state, cleanup, write, rename, or network access is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys

from . import bundle_atomic


SCHEMA = 'crosspascal.retained-stage-readonly-observation'
SCHEMA_VERSION = 1
MAX_PARENT_ENTRIES = 16384
MAX_CANDIDATES = 2048
_SNAPSHOT = re.compile(r'(.+)\.source-freeze-[0-9a-f]{24}\Z')


class UnsafeInspection(ValueError):
    """No partial recovery inventory is authoritative after a refusal."""


def inspect_retained_candidates(parent, *, expected_output=None,
                                max_entries=MAX_PARENT_ENTRIES):
    """Observe exact one-level NTFS stage candidates, including orphan stages.

    Atomic stage names cannot be attributed to a particular output after a
    crash: they remain marked UNATTRIBUTED even when expected_output is set.
    A source snapshot with the matching output prefix is only a name candidate,
    not authenticated as belonging to a specific job.
    """
    if type(max_entries) is not int or not 1 <= max_entries <= MAX_PARENT_ENTRIES:
        raise UnsafeInspection('bounded parent scan limit must be 1..16384')
    try:
        selected_parent = bundle_atomic._local_absolute(parent)
        bundle_atomic._guard_existing_ancestors(selected_parent)
        bundle_atomic._require_ntfs(selected_parent)
        folder = bundle_atomic._guard_stat(selected_parent, directory=True)
        selected_output = None
        if expected_output is not None:
            selected_output = bundle_atomic._local_absolute(expected_output)
            if (os.path.normcase(os.path.dirname(selected_output)) !=
                    os.path.normcase(selected_parent)):
                raise UnsafeInspection(
                    'requested final output must be a direct sibling under parent')
        candidates = []
        with os.scandir(selected_parent) as iterator:
            for checked, entry in enumerate(iterator, 1):
                if checked > max_entries:
                    raise UnsafeInspection(
                        'bounded retained-stage parent inspection entry limit exceeded')
                atomic = bool(bundle_atomic.STAGE_NAME.fullmatch(entry.name))
                snapshot = _SNAPSHOT.fullmatch(entry.name)
                if not atomic and snapshot is None:
                    continue
                if (snapshot is not None and selected_output is not None and
                        snapshot.group(1) != os.path.basename(selected_output)):
                    continue

                try:
                    identity = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise UnsafeInspection(
                        'retained stage candidate changed during inspection') from exc
                if (not stat.S_ISDIR(identity.st_mode) or
                        stat.S_ISLNK(identity.st_mode) or
                        getattr(identity, 'st_file_attributes', 0) &
                        bundle_atomic.REPARSE):
                    raise UnsafeInspection(
                        'unsafe non-directory or linked retained stage candidate')
                candidates.append({
                    'path': os.path.join(selected_parent, entry.name),
                    'kind': ('UNATTRIBUTED_PRIVATE_OUTPUT_STAGE' if atomic
                             else 'OUTPUT_NAMED_SOURCE_SNAPSHOT_CANDIDATE'),
                    'volume_id_observed': identity.st_dev,
                    'file_id_observed': identity.st_ino,
                    'ready': False,
                    'cleanup_allowed': False,
                    'associated_job_proven': False,
                })
                if len(candidates) > MAX_CANDIDATES:
                    raise UnsafeInspection(
                        'retained stage candidate count exceeds bounded report limit')


        after = bundle_atomic._guard_stat(selected_parent, directory=True)
        if (folder.st_dev, folder.st_ino) != (after.st_dev, after.st_ino):
            raise UnsafeInspection('stage parent identity changed during scan')
    except bundle_atomic.UnsafeAtomicBundle as exc:
        raise UnsafeInspection('unsafe NTFS retained stage parent: ' + str(exc)) from exc

    return {
        'schema': SCHEMA,
        'schema_version': SCHEMA_VERSION,
        'status': 'READ_ONLY_UNATTRIBUTED_NOT_RECOVERED',
        'parent': selected_parent,
        'expected_output': selected_output,
        'candidates': sorted(candidates, key=lambda item: item['path']),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Read-only, untrusted retained stage discovery; no READY/cleanup')
    parser.add_argument('--parent', required=True,
                        help='explicit local Windows NTFS parent directory')
    parser.add_argument('--final', default=None,
                        help='optional final output name for source-snapshot filtering')
    parser.add_argument('--max-entries', type=int, default=MAX_PARENT_ENTRIES)
    args = parser.parse_args(argv)
    try:
        result = inspect_retained_candidates(
            args.parent, expected_output=args.final, max_entries=args.max_entries)
    except (UnsafeInspection, OSError, TypeError, ValueError) as exc:
        print('FAILED read-only retained-stage inspection: %s' % exc,
              file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
