import argparse
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def load_image(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, TypeError, ValueError, cv2.error):
        return None


def to_gray(img: "np.ndarray") -> "np.ndarray":
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_roi(img: "np.ndarray", x: int, y: int, w: int, h: int) -> "np.ndarray":
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("ROI requires x/y >= 0 and w/h > 0")
    return img[y:y + h, x:x + w].copy()


def resize_keep(img: "np.ndarray", max_side: int = 640):
    h, w = img.shape[:2]
    if h == 0 or w == 0 or max_side <= 0:
        raise ValueError("Image dimensions and max_side must be positive")
    scale = max_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interpolation)


def draw_marker(img: "np.ndarray", cx: float, cy: float,
                text: Optional[str] = None) -> "np.ndarray":
    out = img.copy()
    center = (int(round(cx)), int(round(cy)))
    cv2.drawMarker(out, center, (0, 0, 255), cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)
    cv2.circle(out, center, 5, (0, 0, 255), -1, cv2.LINE_AA)
    if text is not None:
        cv2.putText(out, text, (center[0] + 15, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 1, cv2.LINE_AA)
    return out


def read_frames(source, n: int = 10) -> List["np.ndarray"]:
    if n <= 0:
        return []
    if isinstance(source, Path):
        source = str(source)
    cap = cv2.VideoCapture(source)
    frames = []
    try:
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def mask_centroid(mask: "np.ndarray") -> Optional[Tuple[float, float]]:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def save_image(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, img)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def self_check():
    with tempfile.TemporaryDirectory(prefix="t05-") as directory:
        root = Path(directory)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[50:150, 100:300] = (255, 0, 0)
        image_path = root / "中文图片.png"
        save_image(image_path, img)
        assert np.array_equal(load_image(str(image_path)), img)
        assert load_image(str(root / "missing.png")) is None
        broken = root / "broken.png"
        broken.write_bytes(b"not an image")
        assert load_image(str(broken)) is None
        print("PASS load_image: Unicode path, missing file, invalid image")

        gray = to_gray(img)
        assert gray.shape == (480, 640) and gray.dtype == np.uint8
        assert gray[60, 110] == 29 and gray[0, 0] == 0
        print("PASS to_gray: BGR -> gray, shape=(480, 640)")

        roi = crop_roi(img, 100, 50, 200, 100)
        assert roi.shape == (100, 200, 3)
        assert np.all(roi == (255, 0, 0))
        print("PASS crop_roi: correct x/y order, shape=(100, 200, 3)")

        assert resize_keep(img, 320).shape == (240, 320, 3)
        assert resize_keep(np.zeros((640, 480, 3), np.uint8), 320).shape == (320, 240, 3)
        assert resize_keep(img, 1280).shape == (960, 1280, 3)
        print("PASS resize_keep: landscape, portrait, downscale and upscale")

        before = img.copy()
        marked = draw_marker(img, 320, 240, "center")
        assert np.array_equal(img, before)
        assert marked.shape == img.shape and not np.array_equal(marked, img)
        assert tuple(marked[240, 320]) == (0, 0, 255)
        print("PASS draw_marker: red point, cross, text; original unchanged")

        video_path = root / "test.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        try:
            if not writer.isOpened():
                raise RuntimeError("MJPG encoder unavailable")
            for value in (30, 100, 200):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        frames = read_frames(video_path, 10)
        assert len(frames) == 3
        assert all(frame.shape == (48, 64, 3) for frame in frames)
        assert all(abs(float(frame.mean()) - value) < 5 for frame, value in zip(frames, (30, 100, 200)))
        assert len(read_frames(video_path, 2)) == 2
        assert read_frames(video_path, 0) == []
        print("PASS read_frames: frame order, count limit, early EOF")

        mask = np.zeros((100, 100), dtype=np.uint8)
        assert mask_centroid(mask) is None
        mask[20:41, 10:31] = 255
        assert mask_centroid(mask) == (20.0, 30.0)
        print("PASS mask_centroid (optional): empty mask and rectangle center")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "截图" / "T05_结果图.png")
    args = parser.parse_args()
    self_check()

    if args.image:
        img = load_image(str(args.image))
        if img is None:
            raise RuntimeError(f"Cannot read image: {args.image}")
    else:
        img = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.rectangle(img, (80, 80), (250, 200), (255, 100, 0), -1)
        cv2.circle(img, (450, 300), 60, (0, 180, 80), -1)
        cv2.putText(img, "OpenCV practice", (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (40, 40, 40), 2, cv2.LINE_AA)

    h, w = img.shape[:2]
    x, y, rw, rh = w // 4, h // 4, max(1, w // 2), max(1, h // 2)
    roi = crop_roi(img, x, y, rw, rh)
    gray = to_gray(img)
    resized = resize_keep(img, 320)
    result = draw_marker(img, (w - 1) / 2, (h - 1) / 2, "image center")
    cv2.rectangle(result, (x, y), (x + rw - 1, y + rh - 1), (0, 180, 0), 2)
    save_image(args.output, result)
    print(f"Image={img.shape}; gray={gray.shape}; ROI={roi.shape}; resized={resized.shape}")

    if args.video:
        frames = read_frames(args.video, 10)
        if not frames:
            raise RuntimeError(f"Cannot read video: {args.video}")
        print(f"PASS supplied video: {len(frames)} frames, first frame={frames[0].shape}")
    print(f"Result saved: {args.output}")
    print("T05 checks completed")


if __name__ == "__main__":
    main()
