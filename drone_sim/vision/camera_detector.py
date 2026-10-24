"""Live camera detector merged from the partner's YOLO-World implementation.

Run: python -m drone_sim.vision.camera_detector
Use --backend heuristic for the supplied NumPy detector (synthetic-image baseline).
Use --deepseek to enable background hazard analysis; press a to analyze a raw
frame, or add --analysis-interval 5 for periodic requests. q/Esc quits; s saves.
YOLO-World setup: https://docs.ultralytics.com/models/yolo-world/
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from drone_sim.configuration.project_paths import output_path
from drone_sim.vision.detector import WireConnectionDetector

WINDOW_TITLE = "Wire / Cable / Plug Detector"
WIRE_PROMPTS = [
    "electrical wire", "copper wire", "wire harness", "electrical cable",
    "battery cable", "power cord", "coaxial cable", "XT60 connector",
    "bullet connector", "JST plug", "charger plug", "USB plug",
    "wall plug", "power adapter", "electrical outlet", "wall power socket",
]
_CLASS_COLOURS = [
    (0, 220, 60), (0, 200, 180), (0, 140, 255), (60, 60, 220),
    (220, 60, 60), (180, 180, 0), (140, 0, 220), (0, 255, 200),
    (255, 180, 0), (0, 180, 255), (80, 220, 80), (220, 80, 200),
    (60, 200, 220), (200, 120, 60), (255, 180, 60), (100, 220, 240),
]


def _auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class YOLOWireDetector:
    """BGR frame adapter returning the same box convention as detector.py.

    bbox_px is (row_min, col_min, row_max, col_max), inclusive; centroid_px
    is (row, col). Model is injectable for tests without downloading weights.
    """
    def __init__(self, checkpoint="yolov8s-worldv2.pt", confidence=0.12,
                 image_size=480, device=None, model=None):
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("Inference image size must be positive.")
        if model is None:
            try:
                from ultralytics import YOLOWorld
            except ImportError:
                raise RuntimeError("YOLO dependencies are missing. Install "
                    "drone_sim/vision/requirements.txt or use --backend heuristic.") from None
            if checkpoint == "yolov8s-worldv2.pt":
                models_directory = Path(__file__).resolve().parent / "models"
                models_directory.mkdir(parents=True, exist_ok=True)
                checkpoint = str(models_directory / checkpoint)
            model = YOLOWorld(checkpoint)
        self.model = model
        self.device = device if device is not None else _auto_device()
        self.confidence, self.image_size = confidence, image_size
        self.model.set_classes(WIRE_PROMPTS)

    def detect_bgr(self, frame):
        prediction = self.model.predict(source=frame, conf=self.confidence,
            imgsz=self.image_size, device=self.device, verbose=False)[0]
        connections, wires = [], []
        boxes = prediction.boxes
        if boxes is not None:
            for xyxy, confidence, class_id in zip(boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
                label = WIRE_PROMPTS[class_id] if 0 <= class_id < len(WIRE_PROMPTS) else str(class_id)
                x0, y0, x1, y1 = xyxy
                row0, col0 = max(0, int(np.floor(y0))), max(0, int(np.floor(x0)))
                row1 = min(frame.shape[0] - 1, int(np.ceil(y1)) - 1)
                col1 = min(frame.shape[1] - 1, int(np.ceil(x1)) - 1)
                if row1 < row0 or col1 < col0:
                    continue
                item = {"type": label, "confidence": float(confidence),
                        "centroid_px": ((row0 + row1) / 2, (col0 + col1) / 2),
                        "bbox_px": (row0, col0, row1, col1),
                        "area_px2": (row1 - row0 + 1) * (col1 - col0 + 1)}
                (wires if any(word in label for word in ("wire", "cable", "cord"))
                 else connections).append(item)
        return {"connections": connections, "wire_segments": wires,
                "wire_count": len(wires), "plug_count": len(connections),
                "overall_score": float(max((v["confidence"] for v in connections + wires), default=0)),
                "warnings": []}


def annotate_frame(frame, detections):
    """Draw cached detections on the CURRENT frame, including skipped frames."""
    image = frame.copy()
    for item in detections.get("connections", []) + detections.get("wire_segments", []):
        row0, col0, row1, col1 = map(int, item["bbox_px"])
        label = item.get("type", item.get("colour", "unknown") + " wire")
        index = WIRE_PROMPTS.index(label) if label in WIRE_PROMPTS else 0
        colour = _CLASS_COLOURS[index]
        cv2.rectangle(image, (col0, row0), (col1, row1), colour, 2)
        if "confidence" in item:
            label += f" {item['confidence']:.2f}"
        cv2.putText(image, label, (col0, max(12, row0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect wires, plugs and outlets in a live camera feed.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("yolo", "heuristic"), default="yolo")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--skip", type=int, default=2, help="Infer every N frames; display current frames.")
    parser.add_argument("--device", default=None, help="cpu | mps | 0 (CUDA)")
    parser.add_argument("--save", type=Path, help="Save annotated video.")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--deepseek", action="store_true", help="Enable a=analyze raw frame.")
    parser.add_argument("--prompt-api-key", action="store_true",
                        help="Ask for a hidden API key if DEEPSEEK_API_KEY is unset.")
    parser.add_argument("--analysis-interval", type=float, default=0,
                        help="Automatic DeepSeek period in seconds; 0 uses the a key.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.conf) or not 0 <= args.conf <= 1 or args.imgsz <= 0 or args.skip < 1:
        parser.error("Require 0 <= conf <= 1, positive imgsz and positive skip.")
    if not np.isfinite(args.analysis_interval) or args.analysis_interval < 0:
        parser.error("Analysis interval must be finite and nonnegative.")
    if args.analysis_interval and not args.deepseek:
        parser.error("--analysis-interval requires --deepseek.")
    if args.prompt_api_key and not args.deepseek:
        parser.error("--prompt-api-key requires --deepseek.")
    return args


def list_cameras():
    found = []
    for index in range(8):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                found.append(index)
        finally:
            capture.release()
    print("Available cameras:", ", ".join(map(str, found)) if found else "none found")


def main(argv=None):
    args = parse_args(argv)
    if args.list_cameras:
        list_cameras()
        return 0

    capture = writer = executor = None
    pending = None
    try:
        analyzer = None
        if args.deepseek:
            from drone_sim.vision.deepseek_analysis import DeepSeekAnalyzer
            api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not api_key and args.prompt_api_key:
                from getpass import getpass
                api_key = getpass("DeepSeek API key (hidden): ").strip()
            analyzer = DeepSeekAnalyzer(api_key = api_key)
        detector = (YOLOWireDetector(args.model, args.conf, args.imgsz, args.device)
                    if args.backend == "yolo" else WireConnectionDetector())
        capture = cv2.VideoCapture(args.camera)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if analyzer is not None:
            executor = ThreadPoolExecutor(max_workers=1)
        frame_index, inference_frame, fps = 0, 0, 0.0
        previous = time.perf_counter()
        last_analysis = -float("inf")
        analysis_status = "DeepSeek: a=analyze" if analyzer else "DeepSeek: disabled"
        analysis_metadata = {}
        detections = {"connections": [], "wire_segments": [], "wire_count": 0, "plug_count": 0}
        print(f"Backend: {args.backend}. Window: {WINDOW_TITLE}")
        print("q/Esc=quit, s=snapshot, a=DeepSeek (requires --deepseek).")

        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames.")
            if frame_index % args.skip == 0:
                if args.backend == "heuristic":
                    # The synthetic baseline is expensive at webcam resolution.
                    scale = min(1.0, args.imgsz / max(frame.shape[:2]))
                    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
                    detections = detector.detect_bgr(small)
                    if scale < 1:
                        for item in detections["connections"] + detections["wire_segments"]:
                            item["bbox_px"] = tuple(int(round(v / scale)) for v in item["bbox_px"])
                            item["centroid_px"] = tuple(v / scale for v in item["centroid_px"])
                            item["area_px2"] = int(round(item["area_px2"] / scale**2))
                            if "length_px" in item:
                                item["length_px"] /= scale
                else:
                    detections = detector.detect_bgr(frame)
                inference_frame = frame_index
            annotated = annotate_frame(frame, detections)
            frame_index += 1

            if pending is not None and pending.done():
                try:
                    result = pending.result()
                    analysis_status = f"DeepSeek frame {analysis_metadata['frame_id']}: {result['risk_level']}"
                    print(json.dumps({**analysis_metadata, "assessment": result}, indent=2), flush=True)
                except Exception as error:
                    analysis_status = "DeepSeek: request failed (see terminal)"
                    print(f"DeepSeek: {error}", file=sys.stderr)
                pending = None

            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            if elapsed > 0:
                instant = 1.0 / elapsed
                fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant
            hud = [f"Wire/Plug Detector {fps:.1f} FPS",
                   f"Wires: {detections['wire_count']}  Plugs/outlets: {detections['plug_count']}"
                   f"  Detection age: {frame_index - inference_frame - 1} frames",
                   analysis_status]
            for index, line in enumerate(hud):
                cv2.putText(annotated, line, (12, 28 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2, cv2.LINE_AA)
            if args.save and writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                        20.0, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot record video to {args.save}.")
            if writer is not None:
                writer.write(annotated)
            cv2.imshow(WINDOW_TITLE, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                prefix = f"camera_snap_{time.time_ns()}"
                for suffix, image in (("_raw.png", frame), (".png", annotated)):
                    if not cv2.imwrite(str(output_path(prefix + suffix)), image):
                        raise RuntimeError("Could not save the camera snapshot.")
                output_path(prefix + ".json").write_text(json.dumps({
                    "frame_id": frame_index, "captured_at": time.time(),
                    "detection_age_frames": frame_index - inference_frame - 1,
                    "detections": detections}, indent=2), encoding="utf-8")
                print(f"Saved {prefix} (raw frame, annotations and detections).")
            automatic = args.analysis_interval > 0 and time.monotonic() - last_analysis >= args.analysis_interval
            if analyzer is not None and pending is None and (key == ord("a") or automatic):
                analysis_metadata = {"frame_id": frame_index, "captured_at": time.time()}
                pending = executor.submit(analyzer.analyze_frame, frame.copy())
                last_analysis = time.monotonic()
                analysis_status = f"DeepSeek: analyzing frame {frame_index}"
        return 0
    except KeyboardInterrupt:
        return 0
    except (ImportError, RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Camera detector: {error}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
