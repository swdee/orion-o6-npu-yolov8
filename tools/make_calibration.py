import argparse
from pathlib import Path
import numpy as np
import cv2

# get file listing from directory
def get_image_files(folder: str,
                    exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
    return [
        str(p) for p in Path(folder).rglob("*")
        if p.suffix.lower() in exts
    ]

def main():
    p = argparse.ArgumentParser(
        description="Build calibration tensor from images"
    )
    p.add_argument(
        "calib_folder",
        help="Path to folder (or file) of raw calibration images"
    )
    p.add_argument(
        "output_npy",
        help="Path where the .npy file will be saved"
    )
    p.add_argument(
        "--size", "-s",
        type=int,
        nargs=2,
        default=(640, 640),
        metavar=("W", "H"),
        help="Target width and height (default: 640 640)"
    )
    args = p.parse_args()

    files = get_image_files(args.calib_folder)
    if not files:
        print(f"No images found in {args.calib_folder!r}")
        return

    tensors = []
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  failed to read {fp!r}, skipping")
            continue

        # resize
        resized = cv2.resize(img, tuple(args.size))
        # normalize
        tensor = resized.astype(np.float32) / 255.0
        # reorder
        tensor = tensor.transpose(2, 0, 1)[None, ...]
        # batch
        tensors.append(tensor)

    data = np.vstack(tensors)
    np.save(args.output_npy, data)
    print(f"Saved calibration data {data.shape} -> {args.output_npy!r}")

if __name__ == "__main__":
    main()
