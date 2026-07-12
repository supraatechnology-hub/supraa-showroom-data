#!/usr/bin/env python3
"""
update_room_status.py — Marks a room's masks ready by calling the backend
internal endpoint (no database credentials in this public repo).
"""
import argparse
import os
import sys
import requests

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--room-id", required=True)
    args = p.parse_args()

    base   = os.environ["BACKEND_URL"].rstrip("/")
    secret = os.environ["INTERNAL_ROOM_SECRET"]
    res = requests.post(
        f"{base}/internal/room-status",
        json={"room_id": args.room_id, "has_masks": True},
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
        timeout=30,
    )
    if res.status_code != 200:
        print(f"[ERROR] room-status update failed: {res.status_code} {res.text}")
        sys.exit(1)
    print(f"[OK] has_masks=true for {args.room_id}")
