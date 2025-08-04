#!/usr/bin/env python3
import os
import json
import base64
import requests
import sys

def load_config(config_path="config.json"):
    if not os.path.isfile(config_path):
        print(f"Error: `{config_path}` not found.", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)

def encode_images(image_paths):
    encoded = []
    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"Warning: image `{img_path}` not found; skipping.", file=sys.stderr)
            continue
        with open(img_path, "rb") as img_f:
            b64 = base64.b64encode(img_f.read()).decode("utf-8")
            encoded.append(b64)
    if not encoded:
        print("Error: No valid images were encoded.", file=sys.stderr)
        sys.exit(1)
    return encoded

def main():
    cfg = load_config()

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "images": encode_images(cfg.get("images", [])),
            "seed": cfg.get("seed", None)
        }
    }

    print("Sending request to:", cfg["endpoint_url"])
    resp = requests.post(cfg["endpoint_url"], headers=headers, json=payload)
    if not resp.ok:
        print(f"Request failed: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    if data.get("status") != "success" or "glb_data" not in data:
        print("Error: Unexpected response:", data, file=sys.stderr)
        sys.exit(1)

    glb_bytes = base64.b64decode(data["glb_data"])
    out_path = cfg.get("output_filename", "output.glb")
    with open(out_path, "wb") as out_f:
        out_f.write(glb_bytes)

    print(f"✅ Saved model to `{out_path}`")

if __name__ == "__main__":
    main()
