import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


PATTERNS = Path.home() / "rc-study/线上自学教案/06_传统视觉/images/patterns.png"
KEYS = {ord("1"): "red", ord("2"): "blue", ord("3"): "green", ord("4"): "yellow"}
RANGES = {"red": [(0, 10), (170, 179)], "blue": [(100, 130)], "green": [(35, 85)], "yellow": [(20, 35)]}
WINDOW = "T07 Color Tracking"


def detect_color(frame, color_name):
    if color_name not in RANGES:
        raise ValueError(f"Unknown color: {color_name}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RANGES[color_name]:
        mask |= cv2.inRange(hsv, (lo, 100, 60), (hi, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    if not contours:
        return None
    moments = cv2.moments(max(contours, key=cv2.contourArea))
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def open_source(src):
    if isinstance(src, str) and src.startswith(("http://", "https://", "rtsp://")):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
        ])
    else:
        cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_image(path):
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"Cannot encode: {path}")
    data.tofile(path)
    print(f"Saved: {path}", flush=True)


def put_text(img, text, pos, scale=0.65):
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - height - 3), (x + width + 2, y + baseline + 2), (255, 255, 255), -1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def mark(img, point, label):
    if point is None:
        return
    x, y = map(round, point)
    cv2.drawMarker(img, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(img, (x, y), 10, (0, 0, 0), 2)
    text = f"{label} ({point[0]:.1f}, {point[1]:.1f})"
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
    put_text(img, text, (max(0, min(x + 15, img.shape[1] - width - 5)), max(25, y - 15)))


def self_test(image_path, output_dir):
    img = read_image(image_path)
    expected = {"red": (480, 270), "blue": (1470, 270), "green": (480, 643.3333), "yellow": (1440, 800)}
    if img.shape[:2] != (1080, 1920):
        raise ValueError("Self-test needs the original 1920x1080 course patterns.png")
    result = img.copy()
    rows = {}
    passed = True
    for color, truth in expected.items():
        detect_color(img, color)
        start = time.perf_counter()
        for _ in range(30):
            point = detect_color(img, color)
        fps = 30 / (time.perf_counter() - start)
        error = float(np.linalg.norm(np.array(point) - truth)) if point is not None else None
        good = error is not None and error <= 15 and fps >= 15
        passed &= good
        rows[color] = {"center": point, "error_px": error, "detection_fps": fps, "passed": good}
        print(f"{'PASS' if good else 'FAIL'} {color}: center={point}, error_px={error}, detection_FPS={fps:.1f}")
        mark(result, point, color)
    blank = np.full((120, 160, 3), 255, np.uint8)
    assert all(detect_color(blank, color) is None for color in RANGES)
    save_image(output_dir / "T07_four_colors.png", result)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "T07_self_test.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("FPS above measures detection only; it excludes reading, drawing and display.")
    if not passed:
        raise SystemExit("T07 self-test failed")
    print("T07 self-test passed: four colors, error <= 15 px, detection FPS >= 15.")


def parse_source(value):
    if value == "usb":
        return 0
    if value.isdecimal():
        return int(value)
    if value.startswith(("http://", "https://", "rtsp://")):
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Source not found: {path}")
    return str(path)


def run(args):
    source = "demo" if args.demo else parse_source(args.source)
    image_mode = isinstance(source, str) and Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    synthetic = args.demo or image_mode
    live = isinstance(source, int) or str(source).startswith(("http://", "https://", "rtsp://"))
    cap = None
    frames = hits = retries = 0
    detection_seconds = 0.0
    result = None
    current = args.color
    frame_period = 1 / 30
    start = None
    try:
        if args.demo:
            base = cv2.resize(read_image(args.patterns), (768, 432))
            source_label = "SYNTHETIC DEMO (course image moved)"
        elif image_mode:
            base = read_image(source)
            source_label = "STILL IMAGE (not video validation)"
        else:
            source_label = "LIVE CAMERA/STREAM" if live else "VIDEO FILE"
            cap = open_source(source)
            if not cap.isOpened() and not live:
                raise RuntimeError(f"Cannot open video: {source}")
            if not live:
                file_fps = cap.get(cv2.CAP_PROP_FPS)
                if np.isfinite(file_fps) and file_fps > 0:
                    frame_period = 1 / file_fps
        if not args.headless:
            fonts = Path('/usr/share/fonts/truetype/dejavu')
            if fonts.is_dir() and not Path(os.environ.get('QT_QPA_FONTDIR', '/missing')).is_dir():
                os.environ['QT_QPA_FONTDIR'] = str(fonts)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1100, 700)
        print(f"Source: {source_label}")
        print("Click the image window: 1/2/3/4 switch colors, s saves a frame, q exits.", flush=True)
        start = time.perf_counter()
        while not args.frames or frames < args.frames:
            iteration_start = time.perf_counter()
            if args.demo:
                frame = np.full((680, 1100, 3), 255, np.uint8)
                x = int(160 + 140 * np.sin(frames * 0.05))
                y = int(170 + 60 * np.cos(frames * 0.05))
                frame[y:y + 432, x:x + 768] = base
            elif image_mode:
                frame = base.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    if not live:
                        print("End of video.")
                        break
                    cap.release()
                    retries += 1
                    if retries > args.retries:
                        raise RuntimeError("Stream unavailable after reconnect attempts")
                    print(f"Stream unavailable; reconnect {retries}/{args.retries}", flush=True)
                    if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    time.sleep(0.2)
                    cap = open_source(source)
                    continue
                retries = 0
            tick = time.perf_counter()
            point = detect_color(frame, current)
            detection_seconds += time.perf_counter() - tick
            frames += 1
            hits += point is not None
            elapsed = time.perf_counter() - start
            fps = frames / max(elapsed, 1e-9)
            result = frame.copy()
            mark(result, point, current)
            put_text(result, f"Tracking: {current} | Loop FPS: {fps:.1f} | Frames: {frames}", (15, 28))
            put_text(result, "Target found" if point else "Target not found", (15, 56))
            put_text(result, source_label, (15, 84), 0.55)
            put_text(result, "1 Red | 2 Blue | 3 Green | 4 Yellow | s Save | q Quit", (15, 112), 0.55)
            if not args.headless:
                cv2.imshow(WINDOW, result)
                delay = 1 if live else max(1, round((frame_period - (time.perf_counter() - iteration_start)) * 1000))
                key = cv2.waitKey(delay) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in KEYS:
                    current = KEYS[key]
                elif key == ord("s"):
                    save_image(args.output_dir / f"T07_{current}_{frames}.png", result)
        elapsed = time.perf_counter() - start
    finally:
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if frames == 0:
        raise RuntimeError("No frames processed; check the source")
    stats = {"source_kind": source_label, "frames": frames, "hits": hits,
             "hit_rate_percent": hits / frames * 100, "elapsed_seconds": elapsed,
             "loop_fps": frames / max(elapsed, 1e-9), "detection_fps": frames / max(detection_seconds, 1e-9)}
    if not synthetic:
        stats["basic_numeric_criteria_pass"] = frames >= 30 and hits / frames >= 0.7 and stats["loop_fps"] >= 10
        print("A hit means a color candidate was found; visually verify that it is the intended target.")
    print(json.dumps(stats, indent=2))
    if synthetic:
        print("Image/demo results are practice only; validate the basic task with a real video or camera.")
    if args.save_last:
        save_image(args.output_dir / "T07_last_frame.png", result)
        (args.output_dir / "T07_run.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="T07 color tracking: image, video, USB camera or stream")
    parser.add_argument("source", nargs="?", default=str(PATTERNS))
    parser.add_argument("--color", choices=RANGES, default="red")
    parser.add_argument("--patterns", type=Path, default=PATTERNS)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "截图")
    args = parser.parse_args()
    if args.frames < 0 or args.retries < 0:
        parser.error("--frames and --retries must be nonnegative")
    if args.headless and not args.frames:
        args.frames = 60
    try:
        if args.self_test:
            self_test(args.patterns, args.output_dir)
        else:
            run(args)
    except KeyboardInterrupt:
        print("Stopped.")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
