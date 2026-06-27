"""
Analitika nad rezultatima trackinga:
  - TrailTracker   : trajektorije (tragovi kretanja) po track ID-u.
  - StatsCollector : agregatna statistika + per-frame CSV log.

Sve radi nad listom detekcija oblika (track_id, class_id, (x1,y1,x2,y2)).
Centroid se računa kao SREDIŠTE DONJEG RUBA boxa — dosljedno ostatku projekta.
"""

import csv
from collections import deque

import cv2

import config as cfg


def _centroid(box):
    """Središte donjeg ruba boxa (x, y2) — gdje vozilo 'dodiruje' cestu."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


class TrailTracker:
    """Pamti i crta zadnjih N pozicija svakog praćenog vozila."""

    def __init__(self, trail_length, thickness):
        self.trail_length = trail_length
        self.thickness = thickness
        self.trails = {}      # {id: deque[(x, y)]}
        self.last_seen = {}   # {id: frame_idx}

    def update(self, frame_idx, detections):
        for track_id, _cls, box in detections:
            if track_id not in self.trails:
                self.trails[track_id] = deque(maxlen=self.trail_length)
            self.trails[track_id].append(_centroid(box))
            self.last_seen[track_id] = frame_idx

    def draw(self, frame, type_of):
        """Nacrtaj putanje. type_of: {id: class_id} za boju po tipu."""
        for track_id, pts in self.trails.items():
            if len(pts) < 2:
                continue
            color = cfg.CLASS_COLORS.get(type_of.get(track_id), cfg.DEFAULT_COLOR)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, self.thickness, cv2.LINE_AA)

    def prune(self, frame_idx):
        """Izbaci ID-eve koji nisu viđeni dulje od trail_length frameova."""
        gone = [tid for tid, seen in self.last_seen.items()
                if frame_idx - seen > self.trail_length]
        for tid in gone:
            self.trails.pop(tid, None)
            self.last_seen.pop(tid, None)


class StatsCollector:
    """Skuplja statistiku kroz video i (opcionalno) zapisuje per-frame CSV."""

    def __init__(self, class_names):
        self.class_names = class_names
        self.unique_ids = set()
        self.id_to_type = {}          # {id: class_id} — zadnji viđeni tip
        self.max_in_frame = 0
        self.rows = []                # per-frame: (frame, time_s, total, *po tipu)

    def update(self, frame_idx, time_s, detections):
        per_type = {c: 0 for c in self.class_names}
        for track_id, class_id, _box in detections:
            self.unique_ids.add(track_id)
            self.id_to_type[track_id] = class_id
            if class_id in per_type:
                per_type[class_id] += 1
        total = sum(per_type.values())
        self.max_in_frame = max(self.max_in_frame, total)
        self.rows.append([frame_idx, round(time_s, 2), total] +
                         [per_type[c] for c in self.class_names])

    def unique_by_type(self):
        counts = {c: 0 for c in self.class_names}
        for class_id in self.id_to_type.values():
            if class_id in counts:
                counts[class_id] += 1
        return counts

    def finalize(self):
        """Zapiši CSV ako je uključeno (SAVE_CSV)."""
        if not cfg.SAVE_CSV:
            return
        header = ["frame", "time_s", "total"] + [self.class_names[c] for c in self.class_names]
        with open(cfg.CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)
        print(f"CSV spremljen: {cfg.CSV_PATH} ({len(self.rows)} redaka)")

    def print_summary(self):
        by_type = self.unique_by_type()
        print("\n===== FINALNA STATISTIKA =====")
        print("Vozila po tipu:")
        for c, name in self.class_names.items():
            print(f"  {name}: {by_type[c]}")
        print(f"Najviše vozila istovremeno u kadru: {self.max_in_frame}")
        print("==============================")
