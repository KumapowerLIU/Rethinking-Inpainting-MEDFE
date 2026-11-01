"""Index an existing image directory for the resize-then-RTV training protocol."""

import argparse
import json
from pathlib import Path

from medfe.data import IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    root = args.images_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--images-root must be a directory")
    files = sorted(
        {
            p.resolve()
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
    )
    if not files:
        raise ValueError("No supported images found")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x") as stream:
        for path in files:
            stream.write(json.dumps({"format": "image", "path": str(path)}) + "\n")
    print(json.dumps({"images": len(files), "manifest": str(args.manifest.resolve())}))


if __name__ == "__main__":
    main()
