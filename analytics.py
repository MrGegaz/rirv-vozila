"""
Analitika nad rezultatima trackinga:
  - TrailTracker   : trajektorije (tragovi kretanja) po track ID-u.
  - StatsCollector : agregatna statistika + per-frame CSV log.

Sve radi nad listom detekcija oblika (track_id, class_id, (x1,y1,x2,y2)).
Centroid se računa kao SREDIŠTE DONJEG RUBA boxa — dosljedno ostatku projekta.
"""

import csv
from collections import deque, Counter

import cv2
import numpy as np

import config as cfg


def _centroid(box):
    """Središte donjeg ruba boxa (x, y2) — gdje vozilo 'dodiruje' cestu."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def classify_color(frame, box):
    """Procijeni dominantnu boju karoserije iz središnjeg dijela boxa.

    Klasičan CV (bez treniranja): uzme se SREDIŠNJI POJAS KAROSERIJE (srednjih
    50% širine, 40–75% visine) da se izbjegnu cesta/pozadina (rubovi), stakla i
    krov (vrh) te kotači/sjena (dno). Uzorak se pretvori u HSV i klasificira:
      - niska zasićenost (S) -> akromatska boja (bijela/siva/crna) po svjetlini V
      - dovoljna zasićenost  -> kromatska boja po tonu (H)
    HSV se koristi jer odvaja TON boje (H) od osvjetljenja (V).
    Vraća (naziv_boje, (H, S, V)) ili (None, None) ako uzorak nije iskoristiv.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return None, None
    cx1 = int(x1 + 0.25 * bw)
    cx2 = int(x2 - 0.25 * bw)
    cy1 = int(y1 + 0.40 * bh)
    cy2 = int(y1 + 0.75 * bh)
    h, w = frame.shape[:2]
    cx1, cx2 = max(0, cx1), min(w, cx2)
    cy1, cy2 = max(0, cy1), min(h, cy2)
    if cx2 - cx1 < 3 or cy2 - cy1 < 3:
        return None, None

    crop = frame[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Medijan je robusniji na pojedine svijetle piksele (refleksije) od srednje vrijednosti.
    H = float(np.median(hsv[:, :, 0]))   # 0..179 (OpenCV)
    S = float(np.median(hsv[:, :, 1]))   # 0..255
    V = float(np.median(hsv[:, :, 2]))   # 0..255
    return _name_from_hsv(H, S, V), (H, S, V)


def _name_from_hsv(H, S, V):
    """Mapiraj izmjerene HSV vrijednosti na naziv boje (vidi pragove u config-u)."""
    # Akromatske boje: niska zasićenost -> razlikuju se samo po svjetlini.
    if S < cfg.COLOR_SAT_MIN:
        if V < cfg.COLOR_VAL_BLACK:
            return "black"
        if V > cfg.COLOR_VAL_WHITE:
            return "white"
        return "gray"

    # Kromatske boje po tonu (H). Vrlo tamno/blijedo padne natrag na akromatsko.
    if V < cfg.COLOR_VAL_BLACK:
        return "black"
    if H < 10 or H >= 160:
        return "red"
    if H < 35:
        return "yellow"
    if H < 85:
        return "green"
    if H < 135:
        return "blue"
    return "red"  # 135–160 (magenta/ružičasto) -> najbliže crvenoj


class ColorVoter:
    """Stabilizira boju po track ID-u glasanjem kroz zadnjih N frameova.

    Pošto već imamo trajne ID-eve, najčešća boja u prozoru sprječava da label
    'treperi' iz framea u frame (sjene, refleksije, promjene osvjetljenja)."""

    def __init__(self, vote_length):
        self.votes = {}       # {id: deque[str]}
        self.last_seen = {}   # {id: frame_idx}
        self.last_hsv = {}    # {id: (H, S, V)} — zadnje izmjereno (za debug)
        self.vote_length = vote_length

    def update(self, frame_idx, frame, detections):
        for track_id, _cls, box in detections:
            color, hsv = classify_color(frame, box)
            if color is None:
                continue
            if track_id not in self.votes:
                self.votes[track_id] = deque(maxlen=self.vote_length)
            self.votes[track_id].append(color)
            self.last_hsv[track_id] = hsv
            self.last_seen[track_id] = frame_idx

    def get(self, track_id):
        """Najčešća boja za ID, ili None ako još nema glasova."""
        dq = self.votes.get(track_id)
        if not dq:
            return None
        return Counter(dq).most_common(1)[0][0]

    def debug_text(self, track_id):
        """'H,S,V' zadnje izmjerenih vrijednosti za ID (za podešavanje pragova)."""
        hsv = self.last_hsv.get(track_id)
        if hsv is None:
            return None
        h, s, v = hsv
        return f"H{int(h)} S{int(s)} V{int(v)}"

    def prune(self, frame_idx):
        gone = [tid for tid, seen in self.last_seen.items()
                if frame_idx - seen > self.vote_length]
        for tid in gone:
            self.votes.pop(tid, None)
            self.last_hsv.pop(tid, None)
            self.last_seen.pop(tid, None)


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
