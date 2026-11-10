# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.


from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys

from . import app_paths, wizard_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local model conversion wizard.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="Serve the wizard without opening a browser automatically.")
    args = parser.parse_args()
    if not (1024 <= args.port <= 65515):
        parser.error("--port must be between 1024 and 65515")
    with wizard_session.instance_lock() as acquired:
        if not acquired:
            wizard_session.open_existing(open_browser=not args.no_browser)
            return 0
        from master import serve
        server = serve(port=args.port, open_browser=not args.no_browser)
        try:
            wizard_session.publish(server)
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
        finally:
            server.server_close()
            wizard_session.clear(server.instance_id)
    return 0


def gui_main() -> int:
    app_paths.prepare()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = app_paths.data_path("logs") / ("wizard-%s-%s.log" % (stamp, os.getpid()))
    original_out, original_err = sys.stdout, sys.stderr
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        try:
            return main()
        except Exception:
            import traceback
            traceback.print_exc()
            if os.name == "nt":
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    None, "The wizard could not start. Details were saved to:\n" + str(log_path),
                    "CrossPascal Convert", 0x10)
            return 1
        finally:
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    raise SystemExit(gui_main())

