"""Middleware entrypoint — start/stop Chroma filesystem vector sync.

Usage:
    python -m dana.middleware.vector_sync_daemon
    python -m dana.middleware.vector_sync_daemon --stop
"""

from __future__ import annotations

import argparse
import time

from dana.memory.vector_sync import start_vector_sync, stop_vector_sync


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChromaDB filesystem vector sync")
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop a previously started in-process syncer (same process only)",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.75,
        help="Debounce window seconds before re-embed/purge",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="If > 0, run then exit (daemon smoke). 0 = run until Ctrl+C.",
    )
    args = parser.parse_args(argv)

    if args.stop:
        print(stop_vector_sync(wait=True))
        return 0

    sync = start_vector_sync(debounce_s=float(args.debounce))
    print(f"vector sync online stats={sync.stats}")
    try:
        if args.run_seconds and args.run_seconds > 0:
            time.sleep(float(args.run_seconds))
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        print(stop_vector_sync(wait=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
