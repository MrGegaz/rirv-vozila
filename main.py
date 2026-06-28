"""
Glavni entry point — live demo: detekcija + praćenje (tracking) vozila.

Pipeline (INFERENCE-ONLY, bez treniranja):
  učitaj predtrenirani YOLO11 -> čitaj video frame-po-frame ->
  detekcija + tracking (ByteTrack) -> trajektorije + statistika ->
  crtanje + overlay -> prikaz uživo.

Kontrole:
  q     = izlaz
  space = pauza / nastavak
  -     = uspori reprodukciju
  +     = ubrzaj reprodukciju

Parametri (model, prag, profil dan/noć, trajektorije, CSV) su u config.py.
"""

import time

import cv2
import torch
from ultralytics import YOLO

import config as cfg
from analytics import TrailTracker, StatsCollector, ColorVoter


def gpu_sanity_check():
    """Ispiši status GPU-a. Ako CUDA nije dostupna, upozori (kriv PyTorch build)."""
    print("CUDA dostupan:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("UPOZORENJE: CUDA nije dostupna -> inferencija ide na CPU (sporo).")
        print("            Provjeriti je li PyTorch instaliran s CUDA buildom (cu121).")


def draw_label(frame, lines, x, y, color):
    """Nacrtaj label (jedan ili više redaka) s podlogom iznad boxa.

    Redci se slažu prema gore: zadnji redak sjedi tik iznad boxa (y), prethodni
    iznad njega — pa je prvi redak na vrhu, a ostali 'ispod' njega.
    """
    if isinstance(lines, str):
        lines = [lines]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    yb = y
    for text in reversed(lines):
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y_top = max(0, yb - th - baseline - 2)
        cv2.rectangle(frame, (x, y_top), (x + tw + 2, yb), color, -1)
        cv2.putText(frame, text, (x + 1, yb - baseline - 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        yb = y_top


def draw_overlay_panel(frame, lines):
    """Poluprozirni panel gore-lijevo sa statistikom trenutnog framea."""
    pad, line_h, width = 8, 22, 230
    height = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    y = pad + 16
    for i, text in enumerate(lines):
        scale = 0.6 if i == 0 else 0.5
        cv2.putText(frame, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h


def main():
    print(f"=== Profil: {cfg.PROFILE} | Video: {cfg.VIDEO_PATH} ===")
    gpu_sanity_check()

    model = YOLO(cfg.MODEL_PATH)

    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    if not cap.isOpened():
        print(f"GREŠKA: ne mogu otvoriti video: {cfg.VIDEO_PATH}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps_in:.1f} FPS")

    trails = TrailTracker(cfg.TRAIL_LENGTH, cfg.TRAIL_THICKNESS)
    stats = StatsCollector(cfg.CLASS_NAMES)
    detect_color = cfg.CFG.get("DETECT_COLOR", False)
    colors = ColorVoter(cfg.COLOR_VOTE_LENGTH) if detect_color else None

    # Granica zone vlastitog vozila (ispod nje se detekcije ignoriraju).
    ignore_y = int(cfg.IGNORE_BOTTOM_REL * height)

    win = "Detekcija i praćenje vozila (q=izlaz, space=pauza, -/+=brzina)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    t_prev = time.time()
    fps_disp = 0.0
    paused = False
    speed_idx = cfg.PLAYBACK_DEFAULT_INDEX
    last_render = None

    def handle_key(key):
        """Obradi tipku. Vrati True ako treba izaći."""
        nonlocal paused, speed_idx
        if key == ord("q"):
            return True
        if key == ord(" "):
            paused = not paused
        elif key == ord("-"):
            speed_idx = max(0, speed_idx - 1)
        elif key == ord("+"):
            speed_idx = min(len(cfg.PLAYBACK_SPEEDS) - 1, speed_idx + 1)
        return False

    while True:
        factor = cfg.PLAYBACK_SPEEDS[speed_idx]
        delay = 1 if factor >= 1.0 else int(40 / factor)

        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Ubrzanje: kod faktora > 1 preskoči obradu/prikaz dijela frameova.
            if factor > 1.0 and (frame_idx % int(factor)) != 0:
                if handle_key(cv2.waitKey(1) & 0xFF):
                    break
                continue

            results = model.track(
                frame, classes=cfg.VEHICLE_CLASSES, conf=cfg.CFG["CONF_THRESHOLD"],
                persist=True, tracker=cfg.TRACKER, device=cfg.DEVICE, imgsz=cfg.IMG_SIZE, verbose=False,
            )

            # Prikupi detekcije (id, class, box) uz filtriranje sitnih boxova.
            detections = []
            r = results[0]
            confs_by_id = {}
            if r.boxes is not None and r.boxes.id is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for box, track_id, class_id, conf in zip(xyxy, ids, clss, confs):
                    x1, y1, x2, y2 = box
                    if (x2 - x1) * (y2 - y1) < cfg.CFG["MIN_BOX_AREA"]:
                        continue
                    # Preskoči zonu vlastitog vozila (hauba na dnu kadra).
                    if (y1 + y2) / 2 > ignore_y:
                        continue
                    detections.append((int(track_id), int(class_id),
                                       (float(x1), float(y1), float(x2), float(y2))))
                    confs_by_id[int(track_id)] = float(conf)

            # Analitika.
            time_s = frame_idx / fps_in
            trails.update(frame_idx, detections)
            stats.update(frame_idx, time_s, detections)
            if colors is not None:
                colors.update(frame_idx, frame, detections)
                colors.prune(frame_idx)

            # Trajektorije ispod boxova (boja po trajno zapamćenom tipu vozila).
            if cfg.SHOW_TRAILS:
                trails.draw(frame, stats.id_to_type)
            trails.prune(frame_idx)

            # Linija vodilja zone vlastitog vozila (za lakše podešavanje).
            if cfg.IGNORE_BOTTOM_REL < 1.0:
                cv2.line(frame, (0, ignore_y), (width, ignore_y), (120, 120, 120), 1)

            # Boxovi + labeli.
            count_by_type = {c: 0 for c in cfg.CLASS_NAMES}
            for tid, cid, (x1, y1, x2, y2) in detections:
                if cid in count_by_type:
                    count_by_type[cid] += 1
                color = cfg.CLASS_COLORS.get(cid, cfg.DEFAULT_COLOR)
                name = cfg.CLASS_NAMES.get(cid, str(cid))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label_lines = [f"ID:{tid} {name} {confs_by_id.get(tid, 0):.2f}"]
                if colors is not None:
                    col = colors.get(tid)
                    if col:
                        label_lines.append(col)
                    if cfg.COLOR_DEBUG:
                        dbg = colors.debug_text(tid)
                        if dbg:
                            label_lines.append(dbg)
                draw_label(frame, label_lines, int(x1), int(y1), color)

            # FPS (eksponencijalno glađen).
            t_now = time.time()
            d = t_now - t_prev
            t_prev = t_now
            if d > 0:
                fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / d)

            # Overlay panel.
            total = sum(count_by_type.values())
            panel = [f"VOZILA U KADRU: {total}"]
            panel += [f"  {cfg.CLASS_NAMES[c]}: {count_by_type[c]}" for c in cfg.CLASS_NAMES]
            panel.append(f"FPS: {fps_disp:.1f}   Brzina: {factor:g}x")
            draw_overlay_panel(frame, panel)

            last_render = frame

        if last_render is not None:
            shown = last_render
            if paused:
                shown = last_render.copy()
                cv2.putText(shown, "PAUZA", (width // 2 - 60, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, shown)

        if handle_key(cv2.waitKey(delay) & 0xFF):
            break

    cap.release()
    cv2.destroyAllWindows()
    stats.finalize()
    stats.print_summary()


if __name__ == "__main__":
    main()
