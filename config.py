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

# Zona vlastitog vozila (hauba / armatura vidljiva na dnu kadra kod dashcama).
# Detekcije čiji je VERTIKALNI CENTAR ispod ove relativne visine se ignoriraju,
# da se vlastiti auto ne detektira kao vozilo. (0..1; 1.0 = isključeno.)
# Podesiti prema tome koliko haube ulazi u kadar (manji broj = veća zona reza).
IGNORE_BOTTOM_REL = 0.85


# --- Profili dan / noć ---
# Mijenja se jednom varijablom; razlikuju se po pragu pouzdanosti i filtriranju
# sitnih detekcija.
PROFILE = "day"  # "day" ili "night"

PROFILES = {
    "day": {
        "CONF_THRESHOLD": 0.40,
        "MIN_BOX_AREA": 1500,
        "DETECT_COLOR": True,      # danju je boja karoserije pouzdana
    },
    "night": {
        "CONF_THRESHOLD": 0.28,    # niži prag — slabije vidljiva vozila (farovi)
        "MIN_BOX_AREA": 2000,      # agresivnije filtriranje refleksija/blještanja
        "DETECT_COLOR": False,     # noću je karoserija u mraku -> boja nepouzdana
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


# --- Prepoznavanje boje vozila ---
# Klasična CV analiza (HSV) središnjeg dijela boxa karoserije. Bez treniranja.
# Uključuje se po profilu (DETECT_COLOR); danju pouzdano, noću isključeno.
# Boja se "glasa" kroz zadnjih N frameova po track ID-u da label ne treperi.
COLOR_VOTE_LENGTH = 15

# Pragovi klasifikacije boje (HSV; H 0-179, S/V 0-255). Podešavaju se brojevima.
# - SAT_MIN: ispod ovog je boja AKROMATSKA (bijela/siva/crna), iznad je kromatska.
# - VAL_WHITE: iznad ovog (uz nisku zasićenost) -> bijela. Spustiti ako se bijeli
#   auti klasificiraju kao 'gray'; podići ako se sivi/srebrni proglašavaju 'white'.
# - VAL_BLACK: ispod ovog -> crna.
COLOR_SAT_MIN = 45
COLOR_VAL_WHITE = 155
COLOR_VAL_BLACK = 60

# Debug: ispiši izmjerene H,S,V uz label da se pragovi gore mogu točno podesiti.
# Pokreni s ovim na True, očitaj S i V na problematičnom (npr. bijelom) vozilu,
# pa namjesti COLOR_VAL_WHITE / COLOR_SAT_MIN i vrati na False.
COLOR_DEBUG = False

# --- Trajektorije (tragovi kretanja) ---
SHOW_TRAILS = True
TRAIL_LENGTH = 20           # koliko zadnjih pozicija (frameova) crtati po vozilu
TRAIL_THICKNESS = 2

# --- Statistika / CSV ---
SAVE_CSV = True
CSV_PATH = "izlaz_statistika.csv"   # per-frame log; sažetak se ispisuje u konzolu

# --- Reprodukcija ---
PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
PLAYBACK_DEFAULT_INDEX = 2           # indeks u PLAYBACK_SPEEDS (2 -> 1.0x)
