#!/usr/bin/env python3
"""
segment_sam3.py — SAM 3 segmentation with Cloudflare R2 storage.
Reads original.jpg from R2, writes masks back to R2.
"""

import argparse
import io
import json
import os
import sys

import boto3
import numpy as np
import requests
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

MIN_COVERAGE_FAIL = 0.1  # % coverage below which a mask is treated as blank

# ── R2 client ─────────────────────────────────────────────────────────────────

def get_r2():
    return boto3.client(
        's3',
        endpoint_url          = os.environ['R2_ENDPOINT'],
        aws_access_key_id     = os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key = os.environ['R2_SECRET_ACCESS_KEY'],
        region_name           = 'auto',
    )

def r2_read_image(r2, bucket, key):
    obj = r2.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(obj['Body'].read())).convert("RGB")

def r2_write_png(r2, bucket, key, array):
    buf = io.BytesIO()
    Image.fromarray(array, mode="L").save(buf, format="PNG")
    buf.seek(0)
    r2.put_object(Bucket=bucket, Key=key, Body=buf, ContentType="image/png")
    print(f"[R2] Uploaded: {key}")

def r2_write_json(r2, bucket, key, data):
    body = json.dumps(data, indent=2).encode()
    r2.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    print(f"[R2] Uploaded: {key}")

def mark_room_ready(room_id):
    base   = os.environ["BACKEND_URL"].rstrip("/")
    secret = os.environ["INTERNAL_ROOM_SECRET"]
    res = requests.post(
        f"{base}/internal/room-status",
        json={"room_id": room_id, "has_masks": True},
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
        timeout=30,
    )
    res.raise_for_status()

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = {
    "kitchen":        {"floor": "kitchen floor tiles",      "wall": "kitchen backsplash wall tiles"},
    "bathroom":       {"floor": "bathroom floor tiles",     "wall": "bathroom wall tiles"},
    "living_room":    {"floor": "living room floor tiles"},
    "bedroom":        {"floor": "bedroom floor tiles"},
    "master_bedroom": {"floor": "master bedroom floor tiles"},
    "dining":         {"floor": "dining room floor tiles"},
    "study":          {"floor": "office floor tiles"},
    "pooja":          {"floor": "pooja room floor tiles",   "wall": "pooja room wall tiles"},
    "toilet":         {"floor": "toilet floor tiles",       "wall": "toilet wall tiles"},
    "shower":         {"floor": "shower floor tiles",       "wall": "shower wall tiles"},
    "utility":        {"floor": "utility room floor tiles", "wall": "utility room wall tiles"},
    "laundry":        {"floor": "laundry room floor tiles"},
    "balcony":        {"floor": "balcony floor tiles outdoor anti-skid"},
    "terrace":        {"floor": "terrace rooftop floor tiles outdoor"},
    "entrance":       {"floor": "entrance foyer floor tiles"},
    "driveway":       {"floor": "driveway parking floor tiles outdoor"},
    "garden_path":    {"floor": "garden pathway stepping stone tiles"},
    "staircase":      {"floor": "staircase step tiles riser tiles"},
}

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print("[SAM3] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    model  = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print(f"[SAM3] Ready on {device}")
    return model, processor, device

def segment_image(image, category, model, processor, device):
    prompts = PROMPTS.get(category, {"floor": "floor tiles"})
    masks   = {}
    for surface, prompt in prompts.items():
        print(f"[SAM3] Segmenting '{surface}': \"{prompt}\"")
        inputs  = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        if len(results["masks"]) > 0:
            combined = results["masks"].any(dim=0).cpu().numpy().astype(np.uint8) * 255
            masks[surface] = combined
            print(f"[SAM3] '{surface}' coverage: {combined.mean() / 255 * 100:.1f}%")
        else:
            print(f"[SAM3] WARNING: no masks for '{surface}'")
            masks[surface] = None
    return masks

# ── Modes ─────────────────────────────────────────────────────────────────────

def run_precompute(room_id, category, model, processor, device):
    r2     = get_r2()
    bucket = os.environ['R2_BUCKET_NAME']
    key    = f"rooms/{room_id}/original.jpg"

    print(f"[PRECOMPUTE] Reading {key} from R2...")
    try:
        image = r2_read_image(r2, bucket, key)
    except Exception as e:
        r2_write_json(r2, bucket, f"rooms/{room_id}/error.json",
                      {"error": f"Could not read image from R2: {e}", "room_id": room_id})
        print(f"[ERROR] {e}")
        sys.exit(1)

    masks = segment_image(image, category, model, processor, device)

    if all(v is None for v in masks.values()):
        r2_write_json(r2, bucket, f"rooms/{room_id}/error.json",
                      {"error": "Segmentation produced no masks.", "room_id": room_id})
        sys.exit(1)

    for surface, arr in masks.items():
        if arr is not None:
            r2_write_png(r2, bucket, f"rooms/{room_id}/{surface}_mask.png", arr)

    print(f"[DONE] Masks for {room_id} written to R2.")


def run_precompute_all(model, processor, device):
    manifest_path = os.path.join(os.path.dirname(__file__), '..', 'rooms_manifest.json')
    with open(manifest_path) as f:
        rooms = json.load(f)

    r2     = get_r2()
    bucket = os.environ['R2_BUCKET_NAME']
    succeeded, failed = [], []

    for entry in rooms:
        room_id, category = entry['room_id'], entry['category']
        print(f"\n=== {room_id} ({category}) ===")
        try:
            key   = f"rooms/{room_id}/original.jpg"
            image = r2_read_image(r2, bucket, key)
            masks = segment_image(image, category, model, processor, device)

            if all(v is None for v in masks.values()):
                raise RuntimeError("segmentation produced no masks")
            blank = [s for s, arr in masks.items()
                     if arr is not None and (arr > 127).sum() / arr.size * 100 < MIN_COVERAGE_FAIL]
            if blank:
                raise RuntimeError(f"blank mask(s): {blank}")

            for surface, arr in masks.items():
                if arr is not None:
                    r2_write_png(r2, bucket, f"rooms/{room_id}/{surface}_mask.png", arr)

            mark_room_ready(room_id)
            print(f"[DONE] {room_id} ready.")
            succeeded.append(room_id)
        except Exception as e:
            print(f"[ERROR] {room_id}: {e}")
            r2_write_json(r2, bucket, f"rooms/{room_id}/error.json", {"error": str(e), "room_id": room_id})
            failed.append(room_id)

    print(f"\n[SUMMARY] {len(succeeded)} succeeded, {len(failed)} failed.")
    if succeeded: print(f"  OK:     {', '.join(succeeded)}")
    if failed:    print(f"  FAILED: {', '.join(failed)}")
    if not succeeded:
        sys.exit(1)


def run_user_upload(image_hash, category, model, processor, device):
    r2     = get_r2()
    bucket = os.environ['R2_BUCKET_NAME']
    key    = f"user_masks/user_{image_hash}/original.jpg"

    print(f"[USER] Reading {key} from R2...")
    try:
        image = r2_read_image(r2, bucket, key)
    except Exception as e:
        r2_write_json(r2, bucket, f"user_masks/user_{image_hash}/error.json",
                      {"error": f"Could not read user image from R2: {e}", "hash": image_hash})
        print(f"[ERROR] {e}")
        sys.exit(1)

    masks = segment_image(image, category, model, processor, device)

    if all(v is None for v in masks.values()):
        r2_write_json(r2, bucket, f"user_masks/user_{image_hash}/error.json",
                      {"error": "Segmentation produced no masks. Try a clearer photo.", "hash": image_hash})
        print(f"[ERROR] No masks produced.")
        sys.exit(1)

    for surface, arr in masks.items():
        if arr is not None:
            r2_write_png(r2, bucket, f"user_masks/user_{image_hash}/{surface}_mask.png", arr)

    print(f"[DONE] User masks written to R2.")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["precompute", "precompute_all", "user_upload"], required=True)
    parser.add_argument("--room-id",     default=None)
    parser.add_argument("--image-hash",  default=None)
    parser.add_argument("--category",    default=None)
    parser.add_argument("--client-slug", default=None)  # passed by workflow, unused in paths
    args = parser.parse_args()

    model, processor, device = load_model()

    if args.mode == "precompute":
        if not args.room_id or not args.category:
            print("ERROR: --room-id and --category required for precompute"); sys.exit(1)
        run_precompute(args.room_id, args.category, model, processor, device)
    elif args.mode == "precompute_all":
        run_precompute_all(model, processor, device)
    else:
        if not args.image_hash or not args.category:
            print("ERROR: --image-hash and --category required for user_upload"); sys.exit(1)
        run_user_upload(args.image_hash, args.category, model, processor, device)
