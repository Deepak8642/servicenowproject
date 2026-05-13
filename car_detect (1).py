"""
VEHICLE COUNTER - Just run it, it works.

INSTALL:  pip install ultralytics opencv-python numpy
RUN:      python car_detect.py --video traffic.mp4
SAVE:     python car_detect.py --video traffic.mp4 --output result.mp4
WEBCAM:   python car_detect.py --camera 0
"""

import argparse, csv, time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Run:  pip install ultralytics opencv-python numpy")

# Vehicle classes to detect
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ══════════════════════════════════════════════════════════════════════════
#  Lightweight IoU tracker  (no extra packages needed)
# ══════════════════════════════════════════════════════════════════════════

class Track:
    _nid = 1
    def __init__(self, box, cls):
        self.id      = Track._nid; Track._nid += 1
        self.box     = np.array(box, dtype=float)
        self.cls     = cls
        self.age     = 1
        self.misses  = 0
        self.counted = False
        self.trail   = deque(maxlen=40)
        cx = int((box[0]+box[2])/2); cy = int((box[1]+box[3])/2)
        self.trail.append((cx, cy))

    def update(self, box):
        self.box   = 0.6*np.array(box) + 0.4*self.box
        self.age  += 1; self.misses = 0
        cx = int((self.box[0]+self.box[2])/2)
        cy = int((self.box[1]+self.box[3])/2)
        self.trail.append((cx, cy))

    @property
    def cx(self): return int((self.box[0]+self.box[2])/2)
    @property
    def cy(self): return int((self.box[1]+self.box[3])/2)


def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua else 0.


class Tracker:
    def __init__(self):
        self.tracks = []

    def step(self, dets):
        used_t = set(); used_d = set()
        for di, d in enumerate(dets):
            best_iou, best_ti = 0.25, -1
            for ti, t in enumerate(self.tracks):
                if ti in used_t: continue
                s = _iou(d[:4], t.box)
                if s > best_iou: best_iou, best_ti = s, ti
            if best_ti >= 0:
                self.tracks[best_ti].update(d[:4])
                used_t.add(best_ti); used_d.add(di)
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(d[:4], d[4]))
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= 10]
        return [t for t in self.tracks if t.age >= 2]


# ══════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ══════════════════════════════════════════════════════════════════════════

def put_label(img, txt, pos, bg, fg=(255,255,255), sc=0.48, th=1):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th2), bl = cv2.getTextSize(txt, f, sc, th)
    x, y = pos
    cv2.rectangle(img, (x-2, y-th2-3), (x+tw+2, y+bl+1), bg, -1)
    cv2.putText(img, txt, (x, y), f, sc, fg, th, cv2.LINE_AA)


def draw_hud(frame, counts, fps, lines):
    H, W = frame.shape[:2]
    total = sum(counts.values())

    # Draw all 3 counting lines
    for i, ly in enumerate(lines):
        thick = 2 if i == 1 else 1
        color = (0,255,255) if i == 1 else (0,180,180)
        cv2.line(frame, (0,ly), (W,ly), color, thick, cv2.LINE_AA)
    cv2.putText(frame, "COUNT LINES", (W//2-60, lines[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

    # Dark stats panel
    rows = 2 + len(counts)
    ph   = 14 + rows*26
    ov   = frame.copy()
    cv2.rectangle(ov, (5,5), (240,ph), (10,10,10), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    y = 30
    cv2.putText(frame, f"TOTAL: {total}", (12,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    y += 28
    for lbl, cnt in sorted(counts.items()):
        cv2.putText(frame, f"  {lbl:<14}: {cnt}", (12,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,220,255), 1, cv2.LINE_AA)
        y += 22
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, y+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1, cv2.LINE_AA)

    ts = datetime.now().strftime("%H:%M:%S")
    put_label(frame, ts, (W-95, 22), bg=(20,20,20), sc=0.45)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def run(source, output_path=None, show=True):
    print("[INFO] Loading YOLOv8n… (downloads ~6MB on first run)")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 3 counting lines at 33%, 50%, 67% of frame height
    # Vehicle counted when it crosses ANY line → works for any camera angle
    LINES = [int(H * f) for f in (0.33, 0.50, 0.67)]
    print(f"[INFO] {W}x{H} @ {FPS:.0f}fps  |  Count lines at y={LINES}")

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))
        print(f"[INFO] Saving output → {output_path}")

    tracker = Tracker()
    counts  = defaultdict(int)
    events  = []

    frame_n = 0
    fps_v = 0.0; fps_t = time.time(); fps_c = 0
    t0 = time.time()

    print("[INFO] Running. Press Q to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_n += 1; fps_c += 1

        # ── Detect ──────────────────────────────────────────────────────
        res = model(frame,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=0.25, iou=0.45,
                    imgsz=640, verbose=False)

        dets = []
        if res and res[0].boxes is not None:
            for box, cid in zip(
                res[0].boxes.xyxy.cpu().numpy(),
                res[0].boxes.cls.cpu().numpy().astype(int)
            ):
                x1,y1,x2,y2 = box.astype(int)
                if (x2-x1)*(y2-y1) > 900:
                    dets.append((x1,y1,x2,y2, cid))

        # ── Track ──────────────────────────────────────────────────────
        active = tracker.step(dets)

        # ── Count + Draw ────────────────────────────────────────────────
        for tr in active:
            x1,y1,x2,y2 = [int(v) for v in tr.box]
            vtype = VEHICLE_CLASSES.get(tr.cls, "Vehicle")

            # Count when centroid crosses any of the 3 lines
            if not tr.counted and len(tr.trail) >= 2:
                prev_cy = tr.trail[-2][1]
                curr_cy = tr.trail[-1][1]
                for ly in LINES:
                    crossed = ((prev_cy < ly <= curr_cy) or
                               (prev_cy > ly >= curr_cy))
                    if crossed and abs(curr_cy - prev_cy) >= 5:
                        tr.counted = True
                        counts[vtype] += 1
                        total = sum(counts.values())
                        ts = datetime.now().strftime("%H:%M:%S")
                        events.append({"time":ts, "frame":frame_n,
                                       "id":tr.id, "type":vtype})
                        print(f"  [{ts}]  {vtype:<12}  id={tr.id:3d}"
                              f"  TOTAL = {total}")
                        break

            # Bounding box
            col = (0,220,80) if tr.counted else (0,140,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            put_label(frame, f"{vtype} #{tr.id}", (x1, y1-2), bg=col)

            # Movement trail
            pts = list(tr.trail)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (80,200,255), 1)
            cv2.circle(frame, (tr.cx, tr.cy), 4, col, -1)

        # ── FPS ─────────────────────────────────────────────────────────
        now = time.time()
        if now - fps_t >= 1.0:
            fps_v = fps_c / (now-fps_t); fps_c = 0; fps_t = now

        draw_hud(frame, counts, fps_v, LINES)

        if writer: writer.write(frame)
        if show:
            cv2.imshow("Vehicle Counter  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    # ── Final report ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total   = sum(counts.values())
    print("\n" + "="*42)
    print("  FINAL VEHICLE COUNT")
    print("="*42)
    print(f"  Total   : {total}")
    for k,v in sorted(counts.items()):
        print(f"    {k:<14}: {v}")
    print(f"  Frames  : {frame_n}")
    print(f"  Time    : {elapsed:.1f}s")
    if elapsed > 0 and total > 0:
        print(f"  Per min : {total/(elapsed/60):.1f}")
    print("="*42)

    if events:
        log = (output_path.rsplit(".",1)[0]+"_log.csv"
               if output_path else "vehicle_log.csv")
        with open(log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=events[0].keys())
            w.writeheader(); w.writerows(events)
        print(f"  Log → {log}")

    return counts


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vehicle Counter")
    g  = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video",   help="Video file path")
    g.add_argument("--camera",  type=int, help="Webcam index (0)")
    ap.add_argument("--output", default=None, help="Save annotated video")
    ap.add_argument("--no-preview", action="store_true", help="No display window")
    args = ap.parse_args()

    run(source      = args.video if args.video else args.camera,
        output_path = args.output,
        show        = not args.no_preview)
