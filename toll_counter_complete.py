"""
╔══════════════════════════════════════════════════════════════════════╗
║         TOLL BOOTH - VEHICLE COUNTER + NUMBER PLATE READER          ║
║                  YOLOv8 + ByteTrack + EasyOCR                       ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALL (run once):
    pip install ultralytics easyocr opencv-python numpy pandas

USAGE:
    # Video file
    python toll_counter_complete.py --video toll.mp4

    # Save annotated output
    python toll_counter_complete.py --video toll.mp4 --output result.mp4

    # Webcam
    python toll_counter_complete.py --camera 0

    # Headless server (no window)
    python toll_counter_complete.py --video toll.mp4 --no-preview

    # Higher accuracy (slower)
    python toll_counter_complete.py --video toll.mp4 --model m
"""

import argparse
import csv
import re
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Optional EasyOCR (graceful fallback if not installed) ──────────────────
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARN] easyocr not installed — plate reading disabled. "
          "Install with:  pip install easyocr")

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not installed. Run:  pip install ultralytics")


# ════════════════════════════════════════════════════════════════
#   CONFIGURATION  ─ tweak these for your specific footage
# ════════════════════════════════════════════════════════════════

# YOLO vehicle classes  {class_id: label}
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Counting line (fraction of frame height, 0=top, 1=bottom)
# Set this INSIDE the toll booth lane where all vehicles must pass
COUNTING_LINE_Y = 0.55          # overrideable via --line flag

# Sensitivity
CONFIDENCE_THRESHOLD   = 0.35   # lower = detect more (may add false positives)
IOU_THRESHOLD          = 0.45   # NMS overlap threshold
MIN_BOX_AREA           = 1500   # ignore tiny detections (shadows, noise)
MIN_CROSSING_DISTANCE  = 12     # pixels of Y-movement to confirm a crossing
TRACK_MEMORY_FRAMES    = 90     # frames to remember a lost track

# Number plate OCR settings
PLATE_CONFIDENCE       = 0.30   # EasyOCR min confidence
PLATE_SCAN_INTERVAL    = 8      # scan for plate every N frames per vehicle
MIN_PLATE_CHARS        = 4      # reject strings shorter than this

# Colours (BGR)
C_COUNTED    = (0,  210,  80)   # green  — already counted vehicle
C_PENDING    = (0, 140, 255)    # orange — detected but not yet crossed line
C_LINE       = (0, 255, 255)    # yellow — counting line
C_PLATE_BOX  = (255,  50, 200)  # magenta — number plate crop

# ════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────
#   NUMBER PLATE DETECTION + OCR
# ─────────────────────────────────────────────────────────────────

class PlateReader:
    """
    Two-stage pipeline:
      1. Locate candidate plate region via heuristic image processing
      2. Run EasyOCR on the candidate crop
    Falls back to full-vehicle-crop OCR if no candidate found.
    """

    def __init__(self):
        if EASYOCR_AVAILABLE:
            print("[INFO] Initialising EasyOCR (first run downloads ~1.5 GB models)…")
            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        else:
            self.reader = None

        # Plate-like aspect ratio bounds (width/height)
        self.ar_min = 1.5
        self.ar_max = 6.0

    # ── Heuristic: find the most plate-like contour inside a vehicle crop ──
    def _find_plate_region(self, crop):
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur  = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

        for c in cnts:
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if h == 0:
                    continue
                ar = w / h
                if self.ar_min <= ar <= self.ar_max and w * h > 800:
                    pad = 4
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(crop.shape[1], x + w + pad)
                    y2 = min(crop.shape[0], y + h + pad)
                    return crop[y1:y2, x1:x2], (x1, y1, x2, y2)
        return None, None

    # ── Preprocess crop for better OCR ────────────────────────────────────
    @staticmethod
    def _preprocess(img):
        if img is None or img.size == 0:
            return None
        # Scale up small crops
        h, w = img.shape[:2]
        scale = max(1, 120 // h)
        if scale > 1:
            img = cv2.resize(img, (w * scale, h * scale),
                             interpolation=cv2.INTER_CUBIC)
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    # ── Clean raw OCR text → keep only plate-like characters ──────────────
    @staticmethod
    def _clean(text):
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        return text

    # ── Public: read plate from a vehicle bounding box crop ───────────────
    def read(self, frame, x1, y1, x2, y2):
        """
        Returns (plate_text, plate_rect_in_frame | None)
        plate_rect = (px1, py1, px2, py2) in FRAME coordinates
        """
        if self.reader is None:
            return "", None

        # Focus on lower half of vehicle (plates are usually there)
        mid_y  = (y1 + y2) // 2
        crop   = frame[mid_y:y2, x1:x2]
        if crop.size == 0:
            return "", None

        # Stage 1 — find plate region
        plate_crop, plate_rect_local = self._find_plate_region(crop)

        # Stage 2 — OCR
        target = plate_crop if plate_crop is not None else crop
        proc   = self._preprocess(target)
        if proc is None:
            return "", None

        try:
            results = self.reader.readtext(proc, detail=1)
        except Exception:
            return "", None

        best_text = ""
        best_conf = 0.0
        for (_, text, conf) in results:
            cleaned = self._clean(text)
            if conf > best_conf and len(cleaned) >= MIN_PLATE_CHARS:
                best_conf = conf
                best_text = cleaned

        if best_conf < PLATE_CONFIDENCE or not best_text:
            return "", None

        # Convert plate rect back to frame coordinates
        frame_rect = None
        if plate_rect_local is not None:
            px1 = x1 + plate_rect_local[0]
            py1 = mid_y + plate_rect_local[1]
            px2 = x1 + plate_rect_local[2]
            py2 = mid_y + plate_rect_local[3]
            frame_rect = (px1, py1, px2, py2)

        return best_text, frame_rect


# ─────────────────────────────────────────────────────────────────
#   DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────

def put_text_with_bg(img, text, pos, font_scale=0.5, color=(255, 255, 255),
                     bg=(30, 30, 30), thickness=1):
    """Draw text with a filled background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


def draw_hud(frame, counts, fps, line_y, elapsed, total_counted):
    h, w = frame.shape[:2]

    # ── Counting line ──────────────────────────────────────────────────────
    cv2.line(frame, (0, line_y), (w, line_y), C_LINE, 2, cv2.LINE_AA)
    cv2.putText(frame, "─── COUNT LINE ───", (w // 2 - 90, line_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_LINE, 1, cv2.LINE_AA)

    # ── Stats panel ────────────────────────────────────────────────────────
    n_rows = 2 + len(counts)
    panel_h = 18 + n_rows * 24 + 10
    panel_w = 240
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 30
    cv2.putText(frame, f"VEHICLES COUNTED: {total_counted}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA)
    y += 24
    for label, cnt in sorted(counts.items()):
        icon = {"car": "Car", "truck": "Truck", "bus": "Bus",
                "motorcycle": "Moto"}.get(label, label.capitalize())
        cv2.putText(frame, f"  {icon}: {cnt}", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (160, 220, 255), 1, cv2.LINE_AA)
        y += 22

    # FPS
    cv2.putText(frame, f"FPS {fps:.1f}  |  {int(elapsed)}s elapsed",
                (16, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (100, 100, 100), 1, cv2.LINE_AA)

    # ── Timestamp ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    put_text_with_bg(frame, ts, (w - 220, 24), font_scale=0.45,
                     color=(200, 255, 200), bg=(20, 20, 20))

    return frame


# ─────────────────────────────────────────────────────────────────
#   MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────────

def run(source, output_path=None, model_size="n", show=True,
        line_fraction=COUNTING_LINE_Y, enable_plates=True):

    global COUNTING_LINE_Y
    COUNTING_LINE_Y = line_fraction

    # ── Models ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLOv8{model_size}  (downloads if first run)…")
    model = YOLO(f"yolov8{model_size}.pt")

    plate_reader = PlateReader() if enable_plates else None

    # ── Video source ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open: {source}")

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SRC_FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    LINE_Y  = int(H * COUNTING_LINE_Y)

    print(f"[INFO] {W}×{H} @ {SRC_FPS:.1f} FPS  |  counting line y={LINE_Y}")

    # ── Output writer ─────────────────────────────────────────────────────
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, SRC_FPS, (W, H))
        print(f"[INFO] Writing → {output_path}")

    # ── State ─────────────────────────────────────────────────────────────
    counts        = defaultdict(int)   # per-class vehicle counts
    counted_ids   = set()              # track IDs already counted
    track_history = defaultdict(lambda: deque(maxlen=40))   # id → centroid trail
    last_seen     = {}                 # id → frame number
    # Per-track: best plate text found so far
    plate_cache   = {}                 # id → {"text": str, "rect": tuple|None}
    plate_scan_at = defaultdict(int)   # id → next frame to scan OCR
    events        = []                 # crossing log

    frame_num = 0
    t_start   = time.time()
    fps_val   = 0.0
    fps_t     = time.time()
    fps_cnt   = 0

    print("[INFO] Running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        fps_cnt   += 1

        # ── YOLO tracking ─────────────────────────────────────────────────
        results = model.track(
            frame,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=640,
        )

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ids   = boxes.id

            if ids is not None:
                xyxys  = boxes.xyxy.cpu().numpy()
                ids_np = ids.cpu().numpy().astype(int)
                cls_np = boxes.cls.cpu().numpy().astype(int)
                conf_np= boxes.conf.cpu().numpy()

                for box, tid, cid, conf in zip(xyxys, ids_np, cls_np, conf_np):
                    x1, y1, x2, y2 = box.astype(int)
                    bw, bh = x2 - x1, y2 - y1

                    # Skip tiny boxes
                    if bw * bh < MIN_BOX_AREA:
                        continue

                    label = VEHICLE_CLASSES.get(cid, "vehicle")
                    cx    = (x1 + x2) // 2
                    cy    = (y1 + y2) // 2

                    last_seen[tid] = frame_num
                    track_history[tid].append((cx, cy))

                    # ── OCR number plate ───────────────────────────────────
                    plate_text = ""
                    plate_rect = None
                    if plate_reader and frame_num >= plate_scan_at[tid]:
                        pt, pr = plate_reader.read(frame, x1, y1, x2, y2)
                        plate_scan_at[tid] = frame_num + PLATE_SCAN_INTERVAL
                        # Keep the longest plate text seen for this track
                        prev = plate_cache.get(tid, {})
                        if len(pt) > len(prev.get("text", "")):
                            plate_cache[tid] = {"text": pt, "rect": pr}

                    cached = plate_cache.get(tid, {})
                    plate_text = cached.get("text", "")
                    plate_rect = cached.get("rect", None)

                    # ── Crossing detection ────────────────────────────────
                    hist = list(track_history[tid])
                    if tid not in counted_ids and len(hist) >= 2:
                        prev_cy = hist[-2][1]
                        curr_cy = hist[-1][1]
                        crossed = (
                            (prev_cy < LINE_Y <= curr_cy) or   # downward
                            (prev_cy > LINE_Y >= curr_cy)      # upward
                        )
                        if crossed and abs(curr_cy - prev_cy) >= MIN_CROSSING_DISTANCE:
                            counted_ids.add(tid)
                            counts[label] += 1
                            direction = "→ entering" if curr_cy >= LINE_Y else "← exiting"
                            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            events.append({
                                "timestamp" : ts,
                                "frame"     : frame_num,
                                "track_id"  : tid,
                                "class"     : label,
                                "direction" : direction,
                                "plate"     : plate_text,
                                "confidence": round(float(conf), 3),
                            })
                            print(f"  [COUNT] {ts}  {label.upper():10s}  "
                                  f"id={tid:4d}  plate={plate_text or '???':12s}  "
                                  f"{direction}  total={len(counted_ids)}")

                    # ── Draw bounding box ─────────────────────────────────
                    is_counted = tid in counted_ids
                    color = C_COUNTED if is_counted else C_PENDING
                    thickness = 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    # Label above box
                    label_str = f"{label}#{tid}  {conf:.2f}"
                    put_text_with_bg(frame, label_str, (x1, y1 - 2),
                                     font_scale=0.46, color=(255, 255, 255),
                                     bg=color, thickness=1)

                    # Plate text below box
                    if plate_text:
                        put_text_with_bg(frame, f" {plate_text} ",
                                         (x1, y2 + 16),
                                         font_scale=0.52,
                                         color=(255, 255, 255),
                                         bg=C_PLATE_BOX, thickness=1)

                    # Draw plate rect if found
                    if plate_rect:
                        px1, py1_, px2, py2_ = plate_rect
                        cv2.rectangle(frame,
                                      (px1, py1_), (px2, py2_),
                                      C_PLATE_BOX, 2)

                    # Centroid trail
                    pts = list(track_history[tid])
                    for i in range(1, len(pts)):
                        alpha = int(180 * i / len(pts))
                        cv2.line(frame, pts[i - 1], pts[i],
                                 (alpha, 200, 255 - alpha), 1)

        # ── Purge stale tracks ────────────────────────────────────────────
        stale = [tid for tid, fn in last_seen.items()
                 if frame_num - fn > TRACK_MEMORY_FRAMES]
        for tid in stale:
            last_seen.pop(tid, None)
            plate_scan_at.pop(tid, None)
            # Keep plate_cache so we can still write it to the log

        # ── FPS ───────────────────────────────────────────────────────────
        now = time.time()
        if now - fps_t >= 1.0:
            fps_val = fps_cnt / (now - fps_t)
            fps_cnt = 0
            fps_t   = now

        # ── HUD overlay ───────────────────────────────────────────────────
        elapsed = time.time() - t_start
        frame   = draw_hud(frame, counts, fps_val, LINE_Y,
                           elapsed, len(counted_ids))

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("Toll Vehicle Counter  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit by user.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Final report ──────────────────────────────────────────────────────
    total    = len(counted_ids)
    elapsed  = time.time() - t_start
    plates   = [e["plate"] for e in events if e["plate"]]

    print("\n" + "═" * 55)
    print("  TOLL BOOTH  —  FINAL REPORT")
    print("═" * 55)
    print(f"  Total vehicles counted : {total}")
    for lbl, cnt in sorted(counts.items()):
        print(f"    {lbl.capitalize():<14}: {cnt}")
    print(f"  Plates extracted       : {len(plates)}/{total}")
    for e in events:
        if e["plate"]:
            print(f"    {e['class']:<10} → {e['plate']}")
    print(f"  Frames processed       : {frame_num}")
    print(f"  Duration               : {elapsed:.1f}s")
    if elapsed > 0 and total > 0:
        print(f"  Vehicles / minute      : {total / (elapsed/60):.1f}")
    print("═" * 55)

    # ── Save CSV ──────────────────────────────────────────────────────────
    if events:
        stem     = Path(output_path).stem if output_path else "vehicle_count"
        log_path = stem + "_log.csv"
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=events[0].keys())
            w.writeheader()
            w.writerows(events)
        print(f"\n[INFO] Log saved → {log_path}")

    return counts, events


# ════════════════════════════════════════════════════════════════
#   CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Toll Booth — Vehicle Counter + Plate Reader  (YOLOv8 + EasyOCR)"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  help="Path to video file (mp4 / avi / mov / mkv)")
    src.add_argument("--camera", type=int,
                     help="Camera index (0 = default webcam)")

    ap.add_argument("--output",  default=None,
                    help="Save annotated video to this path (optional)")
    ap.add_argument("--model",   default="n",
                    choices=["n", "s", "m", "l", "x"],
                    help="YOLOv8 size: n=nano (fast) … x=extra-large (accurate). Default: n")
    ap.add_argument("--line",    type=float, default=COUNTING_LINE_Y,
                    help=f"Counting line as fraction of frame height (default {COUNTING_LINE_Y})")
    ap.add_argument("--no-preview", action="store_true",
                    help="Disable live window (headless servers)")
    ap.add_argument("--no-plates",  action="store_true",
                    help="Skip number plate OCR (faster)")

    args = ap.parse_args()
    source = args.video if args.video else args.camera

    run(
        source        = source,
        output_path   = args.output,
        model_size    = args.model,
        show          = not args.no_preview,
        line_fraction = args.line,
        enable_plates = not args.no_plates,
    )
