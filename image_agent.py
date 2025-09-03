import os
import re
import io
import sys
import time
import json
import base64
import pathlib
from typing import Dict, List, Tuple

# -------------------------
# Config via ENV VARS
# -------------------------
# PROVIDER: "openai" or "auto1111"
PROVIDER = os.getenv("IMG_PROVIDER", "openai").lower()

# Output directory
OUT_DIR = os.getenv("IMG_OUT_DIR", "out_images")

# Cooldown seconds between image generations (provider rate limits)
COOLDOWN_SEC = float(os.getenv("IMG_COOLDOWN_SEC", "2"))

# Hard safety limit: stop if > 50 images would be created
HARD_LIMIT = int(os.getenv("IMG_HARD_LIMIT", "50"))

# OpenAI config (if PROVIDER=openai)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1536x1024")  # closest to 16:9

# AUTOMATIC1111 config (if PROVIDER=auto1111)
A1111_BASE = os.getenv("A1111_BASE", "http://127.0.0.1:7860")
A1111_WIDTH = int(os.getenv("A1111_WIDTH", "1024"))
A1111_HEIGHT = int(os.getenv("A1111_HEIGHT", "576"))  # 16:9
A1111_STEPS = int(os.getenv("A1111_STEPS", "28"))
A1111_SAMPLER = os.getenv("A1111_SAMPLER", "DPM++ 2M Karras")
A1111_NEG = os.getenv(
    "A1111_NEG",
    "blurry, low quality, extra fingers, deformed hands, text artifacts, watermark, logo"
)

# -------------------------
# Prompt file parser
# -------------------------
SCENE_RE = re.compile(r"^\s*SCENE\s+(\d+)\s*:\s*$", re.IGNORECASE)
COMMON_RE = re.compile(r"^\s*COMMON\s+STYLE\s*:\s*$", re.IGNORECASE)

def parse_prompt_file(path: str) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Returns: (common_style, [(scene_number, scene_prompt), ...])
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    common_style = ""
    scenes: List[Tuple[int, str]] = []

    i = 0
    N = len(lines)
    state = "idle"
    buf: List[str] = []
    current_scene = None

    while i < N:
        line = lines[i]

        if COMMON_RE.match(line):
            # flush previous scene if any
            if current_scene is not None and buf:
                scenes.append((current_scene, "\n".join(buf).strip()))
                buf = []
                current_scene = None
            state = "common"
            i += 1
            continue

        m = SCENE_RE.match(line)
        if m:
            # flush common buffer if active
            if state == "common":
                common_style = "\n".join(buf).strip()
                buf = []
                state = "idle"

            # flush previous scene buffer
            if current_scene is not None and buf:
                scenes.append((current_scene, "\n".join(buf).strip()))
                buf = []

            current_scene = int(m.group(1))
            state = "scene"
            i += 1
            continue

        # Accumulate
        buf.append(line)
        i += 1

    # Tail flush
    if state == "common":
        common_style = "\n".join(buf).strip()
    elif current_scene is not None:
        scenes.append((current_scene, "\n".join(buf).strip()))

    # sort scenes by number
    scenes.sort(key=lambda x: x[0])
    return common_style, scenes

# -------------------------
# Providers
# -------------------------
class ProviderBase:
    def generate(self, prompt: str, n: int) -> List[bytes]:
        raise NotImplementedError

class OpenAIProvider(ProviderBase):
    def __init__(self, api_key: str, size: str):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Install openai>=1.0.0: pip install openai") from e
        self.client = OpenAI(api_key=api_key)
        self.size = size

    def generate(self, prompt: str, n: int) -> List[bytes]:
        # OpenAI Images API returns base64; request n images
        resp = self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=self.size,
            n=n,
        )
        out = []
        for d in resp.data:
            b64 = d.b64_json
            out.append(base64.b64decode(b64))
        return out

class Auto1111Provider(ProviderBase):
    def __init__(self, base: str):
        import requests  # local use only
        self.base = base
        self.session = requests.Session()

    def generate(self, prompt: str, n: int) -> List[bytes]:
        import requests
        url = f"{self.base}/sdapi/v1/txt2img"
        payload = {
            "prompt": prompt,
            "negative_prompt": A1111_NEG,
            "steps": A1111_STEPS,
            "sampler_name": A1111_SAMPLER,
            "width": A1111_WIDTH,
            "height": A1111_HEIGHT,
            "batch_size": min(n, 2),  # generate up to 2 per call
        }
        r = self.session.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        imgs = []
        for im_b64 in data.get("images", []):
            imgs.append(base64.b64decode(im_b64))
        # If n>batch_size, call again
        if len(imgs) < n:
            imgs += self.generate(prompt, n - len(imgs))
        return imgs[:n]

def build_provider() -> ProviderBase:
    if PROVIDER == "openai":
        return OpenAIProvider(api_key=OPENAI_API_KEY, size=OPENAI_IMAGE_SIZE)
    elif PROVIDER == "auto1111":
        return Auto1111Provider(base=A1111_BASE)
    else:
        raise RuntimeError(f"Unknown provider: {PROVIDER}")

# -------------------------
# Runner
# -------------------------
def ensure_out_dir():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s.strip())
    return s[:80] if len(s) > 80 else s

def main(prompt_file: str):
    ensure_out_dir()
    common_style, scenes = parse_prompt_file(prompt_file)

    if not scenes:
        print("No scenes found. Ensure your file has 'SCENE X:' entries.", file=sys.stderr)
        sys.exit(1)

    total_requested = len(scenes) * 2
    if total_requested > HARD_LIMIT:
        print(f"Requested {total_requested} images (> {HARD_LIMIT}). Aborting.", file=sys.stderr)
        sys.exit(1)

    provider = build_provider()

    print(f"[INFO] Provider: {PROVIDER}")
    print(f"[INFO] Scenes: {len(scenes)} | Will generate {total_requested} images")
    if common_style:
        print("[INFO] Common style present and will be applied.")

    generated = 0
    for scene_num, scene_prompt in scenes:
        merged_prompt = scene_prompt.strip()
        if common_style:
            # Apply style by appending; you can also prepend if preferred.
            merged_prompt = f"{scene_prompt.strip()}\n\nStyle:\n{common_style.strip()}"

        # Exactly 2 images per scene
        try:
            images = provider.generate(merged_prompt, n=2)
        except Exception as e:
            print(f"[ERROR] Generation failed for scene {scene_num}: {e}", file=sys.stderr)
            sys.exit(2)

        for idx, img_bytes in enumerate(images, start=1):
            fname = f"scene_{scene_num:02d}_{idx}.png"
            out_path = os.path.join(OUT_DIR, sanitize_filename(fname))
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"[OK] Saved {out_path}")
            generated += 1

        # cooldown between scenes (helps providers with rate limits)
        if COOLDOWN_SEC > 0:
            time.sleep(COOLDOWN_SEC)

    print(f"[DONE] Generated {generated} images into {OUT_DIR}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_agent.py prompts.txt", file=sys.stderr)
        sys.exit(64)
    main(sys.argv[1])
