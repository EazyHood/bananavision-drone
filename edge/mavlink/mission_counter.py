#!/usr/bin/env python3
"""MAVLink companion-computer folder bridge.

This template listens to a camera directory, runs BananaVision on new images,
and writes the same restart-safe mission outputs as `bananavision mission-watch`.
Mission-specific MAVLink telemetry plumbing is left explicit because autopilot
setups differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bananavision.mission_runner import watch_mission
from bananavision.pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/mission"))
    parser.add_argument("--config", type=Path, default=Path("configs/banana_uav.yaml"))
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--once", action="store_true", help="Process currently ready images and exit.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior mission_watch_state.json.")
    args = parser.parse_args()

    manifest = watch_mission(
        args.watch,
        args.output,
        load_config(args.config),
        poll_interval=args.interval,
        settle_seconds=args.settle_seconds,
        max_cycles=1 if args.once else None,
        resume=not args.no_resume,
    )
    print(
        f"{manifest['image_count']} image(s), "
        f"{manifest['total_detections']} banana candidates, "
        f"{manifest['failure_count']} failure(s)"
    )


if __name__ == "__main__":
    main()
