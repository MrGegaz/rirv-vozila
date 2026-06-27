"""
Konfiguracija sustava za detekciju i praćenje (tracking) vozila.

Svi parametri koji se mogu eksperimentalno podešavati nalaze se ovdje, na jednom
mjestu, kako bi se sustav lako kalibrirao na konkretnom videu (dnevni / noćni).

Sustav je INFERENCE-ONLY: koristi se isključivo predtrenirani COCO model.
Nema treniranja, fine-tuninga ni stvaranja dataseta.
"""

import torch

# --- Osnovne putanje i model ---
# Putanja do videa nad kojim se pokreće demo. Promijeniti na noćni video za drugi demo.
VIDEO_PATH = "video/DayDrive1.mp4"

# Predtrenirani YOLO11 model (small). Preuzima se automatski pri prvom pokretanju.
# Probati i "yolo11m.pt" ako "s" premalo hvata sitna/tamna vozila (uz pad FPS-a).
MODEL_PATH = "yolo11s.pt"

# Uređaj za inferenciju: 0 = prvi GPU (RTX 3050), inače CPU.
# Eksplicitno postavljeno da ultralytics ne padne tiho na CPU.
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# COCO indeksi klasa vozila: car, motorcycle, bus, truck.
VEHICLE_CLASSES = [2, 3, 5, 7]

# Ulazna rezolucija modela. Smanjiti (npr. 480) ako pada FPS.
IMG_SIZE = 640

# Tracker. ByteTrack se bolje nosi s pokretnom kamerom od defaulta.
# Fallback: "botsort.yaml" ako bytetrack ne radi.
TRACKER = "bytetrack.yaml"

# Procesiraj svaki N-ti frame radi performansi (1 = svaki frame).
FRAME_SKIP = 1


# --- Profili dan / noć ---
# Mijenja se jednom varijablom; razlikuju se po pragu pouzdanosti, filtriranju
# sitnih detekcija i (opcionalno) pojačanju kontrasta.
PROFILE = "day"  # "day" ili "night"

PROFILES = {
    "day": {
        "CONF_THRESHOLD": 0.40,
        "MIN_BOX_AREA": 1500,
        "USE_CLAHE": False,
    },
    "night": {
        "CONF_THRESHOLD": 0.28,    # niži prag — slabije vidljiva vozila (farovi)
        "MIN_BOX_AREA": 2000,      # agresivnije filtriranje refleksija/blještanja
        "USE_CLAHE": False,        # uključiti SAMO ako gola detekcija noću podbaci
    },
}

# Aktivni profil — uvozi se u ostatak koda kao CFG.
CFG = PROFILES[PROFILE]


# --- Nazivi i boje klasa (BGR, jer OpenCV) ---
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

CLASS_COLORS = {
    2: (0, 255, 0),      # car        -> zelena
    3: (0, 255, 255),    # motorcycle -> žuta
    5: (255, 0, 0),      # bus        -> plava
    7: (0, 165, 255),    # truck      -> narančasta
}

# Rezervna boja za eventualnu nepoznatu klasu.
DEFAULT_COLOR = (200, 200, 200)


# --- Trajektorije (tragovi kretanja) ---
SHOW_TRAILS = True
TRAIL_LENGTH = 30           # koliko zadnjih pozicija (frameova) crtati po vozilu
TRAIL_THICKNESS = 2

# --- Statistika / CSV ---
SAVE_CSV = True
CSV_PATH = "izlaz_statistika.csv"   # per-frame log; sažetak se ispisuje u konzolu

# --- Reprodukcija ---
PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
PLAYBACK_DEFAULT_INDEX = 2           # indeks u PLAYBACK_SPEEDS (2 -> 1.0x)
