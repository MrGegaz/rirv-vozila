# Plan: Detekcija i analiza vozila (Računarski i robotski vid)

## Cilj projekta
Live demo nad videima **gradske vožnje (užim centrom grada), snimljeno iz perspektive vozila u pokretu** (dashcam-style, kamera se kreće). Postoje **dva videa: dnevna vožnja i noćna vožnja** — rješenje mora raditi na oba. Sustav u stvarnom vremenu treba:

1. **Detektirati vozila** (auto, kamion, autobus, motocikl)
2. **Pratiti (tracking)** ista vozila kroz frameove uz stabilan ID
3. **Brojati vozila** preko virtualne linije
4. **Klasificirati tip vozila** i voditi statistiku po tipu

> **VAŽNA NAPOMENA O VIDEU:** Kamera je **u pokretu** (snimka iz vozila), nije fiksna nadzorna kamera. To znači da se cijela scena pomiče, parkirana vozila "putuju" kroz kadar, i pozadina nije statična. Plan je tome prilagođen (vidi sekciju "Specifičnosti pokretne kamere"). Brojanje preko linije ovdje funkcionira drugačije nego kod fiksne kamere i treba ga interpretirati kao "broj jedinstvenih vozila koja su prošla kroz kadar / prešla liniju", ne kao prometni protok.

> **🚫 STROGO PRAVILO — PROFESOROVI VIDEI SU SAMO ZA DEMO:** Profesorovi videi (dnevni i noćni) služe **isključivo za live demo / evaluaciju** — pokretanje gotovog sustava uživo da se vidi kako radi. **NIKAKO se ne smiju koristiti za treniranje, fine-tuning, niti kao izvor trening podataka** (npr. izvlačenje frameova, anotiranje, augmentacija). Model ostaje na **predtreniranim COCO težinama** (`yolo11s.pt` / `yolo11m.pt` kakvi se preuzmu). Cijeli pipeline je **inference-only**: učitaj predtrenirani model → pokreni na videu → prikaži rezultate. Agent ne smije pisati nikakav training/fine-tuning kod, ne smije zvati `model.train()`, niti stvarati dataset iz ovih videa.

---

## Tehnološki stack
- **Python 3.10+**
- **ultralytics** (YOLO11 / YOLOv8) — detekcija + ugrađeni tracking
- **OpenCV** (`opencv-python`) — čitanje videa, crtanje, prikaz
- **numpy**
- **PyTorch s CUDA podrškom** (obavezno za GPU ubrzanje)

### Hardver: NVIDIA RTX 3050 (6 GB VRAM)
GPU je dostupan → koristiti CUDA. 6 GB je dovoljno za real-time inferenciju srednjih modela.
- **Preporučeni model: `yolo11s.pt`** (small) — najbolji omjer točnosti i brzine na 3050. Nano (`n`) je nepotrebno štedljiv kad imaš GPU; small daje osjetno bolju detekciju (važno za noćni video).
- `yolo11m.pt` (medium) je također izvediv real-time na 3050 ako se pokaže da `s` premalo hvata sitna/tamna vozila — testirati oba i izabrati prema FPS-u.
- **Eksplicitno postaviti `device=0`** (prvi GPU) u svim `model.track()` pozivima, inače ultralytics zna pasti na CPU.

```
# Instalacija PyTorch s CUDA (provjeriti aktualnu verziju na pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy
```

**Provjera da GPU stvarno radi (agent neka ubaci na početak):**
```python
import torch
print("CUDA dostupan:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nema")
```
Ako `cuda.is_available()` vrati `False`, PyTorch je instaliran bez CUDA buildova — reinstalirati s ispravnim `--index-url`.

---

## COCO klase vozila (relevantni indeksi)
- `2` = car
- `3` = motorcycle
- `5` = bus
- `7` = truck

Tracking i detekciju ograničiti na ove klase: `classes=[2, 3, 5, 7]`.

---

## Struktura projekta
```
projekt/
├── main.py              # glavni entry point, live demo loop
├── tracker_logic.py     # logika brojanja preko linije + statistika
├── config.py            # parametri (putanje, prag, pozicija linije, boje)
├── requirements.txt
├── video/
│   ├── gradska_voznja_dan.mp4    # dnevni video
│   └── gradska_voznja_noc.mp4    # noćni video
└── README.md
```

---

## Funkcionalni zahtjevi (što kod mora raditi)

### 1. Učitavanje i obrada videa
- Učitati video preko `cv2.VideoCapture`.
- Petlja frame-po-frame do kraja videa.
- Prikaz uživo u prozoru (`cv2.imshow`), izlaz na tipku `q`.
- Dohvatiti FPS i rezoluciju iz videa (`cap.get(cv2.CAP_PROP_FPS)` itd.) i koristiti ih za skaliranje overlaya.

### 2. Detekcija + tracking
- Koristiti `model.track(frame, classes=[2,3,5,7], persist=True, tracker="bytetrack.yaml", device=0)`.
- `persist=True` je **obavezno** da ID-evi ostaju stabilni kroz frameove.
- `device=0` da inferencija ide na RTX 3050, a ne na CPU.
- ByteTrack se bolje nosi s pokretnom kamerom od defaultnog trackera — preferirati `bytetrack.yaml`.
- Iz rezultata izvući: bounding box (xyxy), track ID, class ID, confidence.
- Postaviti prag pouzdanosti (npr. `conf=0.4`) da se smanje lažne detekcije zbog trešnje kamere.

### 3. Crtanje na frameu
- Za svako vozilo nacrtati bounding box.
- **Boja okvira ovisi o tipu vozila** (npr. car=zelena, truck=narančasta, bus=plava, motorcycle=žuta).
- Label iznad okvira: `ID:{id} {tip} {conf}` (npr. `ID:14 car 0.82`).
- Nacrtati virtualnu liniju brojanja preko kadra.
- Overlay panel (gore lijevo) sa statistikom: ukupan broj, broj po tipu, broj vozila trenutno u kadru.

### 4. Brojanje preko linije
- Definirati virtualnu liniju (horizontalnu ili dijagonalnu) u `config.py` kao dvije točke ili kao y-koordinatu.
- Pratiti **centroid** (središte donjeg ruba boxa je stabilnije od centra) svakog track ID-a.
- Spremati prethodnu poziciju centroida po ID-u.
- Kad centroid **prijeđe liniju** (promjena predznaka udaljenosti od linije između dva framea) → inkrementiraj brojač **i zabilježi ID kao "izbrojan"** da se ne broji dvaput.
- Voditi zasebne brojače po tipu vozila.

### 5. Statistika
- `total_count` — ukupno prešlo liniju
- `count_by_type` — dict {car: n, truck: n, bus: n, motorcycle: n}
- `current_in_frame` — broj aktivnih track ID-eva u trenutnom frameu
- Prikazati sve na overlay panelu.
- (Opcionalno) na kraju videa ispisati finalnu statistiku u konzolu.

---

## Specifičnosti pokretne kamere (KRITIČNO — prilagodbe za ovaj video)

Standardni "brojanje preko linije" tutorijali pretpostavljaju **fiksnu** kameru. Ovdje kamera je u vozilu, pa treba sljedeće prilagodbe:

1. **Pozicija linije:** Postaviti liniju u donju trećinu kadra (gdje su vozila najbliža i najveća), ili kao dijagonalu prema točki nestajanja. Vozila koja se mimoilaze prolaze rubovima kadra — razmisliti o **dvije linije** (lijevi rub = nadolazeći promet, desni rub / centar = vozila ispred). Ostaviti poziciju lako podesivom u `config.py` jer će se kalibrirati eksperimentalno na konkretnom videu.

2. **Veća tolerancija na izgubljene/ponovno nađene ID-eve:** Zbog trešnje i okluzija (vozila iza drugih vozila, stupova) tracker povremeno izgubi i ponovno dodijeli ID. Koristiti ByteTrack i ne paničariti oko ID switcheva — za demo je prihvatljivo.

3. **Filtriranje po veličini boxa (opcionalno):** Vrlo mali boxovi u daljini (parkirana vozila daleko, vozila preko raskrižja) generiraju šum. Može se ignorirati detekcije ispod neke minimalne površine boxa da brojanje bude stabilnije.

4. **Interpretacija rezultata:** U prezentaciji jasno reći da brojač predstavlja "broj jedinstvenih vozila detektiranih/prošlih kroz scenu", a ne prometni protok kroz fiksnu točku ceste. To je iskrena i točna interpretacija za pokretnu kameru.

5. **Performanse:** Gradska scena ima puno vozila istovremeno → uz GPU (3050) koristiti `s` ili `m` model. Ako pada FPS, smanjiti `imgsz` ili procesirati svaki drugi frame.

---

## Specifičnosti noćne vožnje (KRITIČNO — drugi video je noćni)

Noćni video je znatno teži za detekciju od dnevnog. Mora se eksplicitno testirati i podesiti zasebno:

1. **Slabija detekcija u mraku:** Vozila su često vidljiva samo kao parovi farova/stražnjih svjetala, bez jasne siluete. YOLO trenirana na COCO-u detektira slabije po noći. Ovdje `s`/`m` model (umjesto `n`) stvarno pomaže — vrijedi žrtvovati nešto FPS-a za bolju detekciju. Razmotriti **niži `conf` prag noću** (npr. 0.25–0.30 umjesto 0.4) da se uhvate slabije vidljiva vozila, uz rizik nešto više lažnih detekcija.

2. **Bliještanje farova i refleksije:** Jaki farovi nadolazećih vozila i refleksije na mokrom asfaltu stvaraju svijetle mrlje koje znaju zbuniti detektor i tracker. Filtriranje sitnih detekcija (`MIN_BOX_AREA`) ovdje pomaže protiv lažnih okvira na refleksijama.

3. **Zaseban config po videu:** Ne forsirati iste parametre na oba videa. Napraviti **dva profila u `config.py`** (npr. `PROFILE = "day"` / `"night"`) s različitim `CONF_THRESHOLD`, eventualno `MIN_BOX_AREA` i pozicijom linije. Agent neka strukturira config tako da se profil mijenja jednom varijablom.

4. **Opcionalno poboljšanje slike (samo ako detekcija noću podbaci):** Lagani pretprocessing framea prije detekcije — npr. CLAHE (`cv2.createCLAHE`) za izjednačavanje kontrasta ili gamma korekcija da se posvijetle tamna područja. **Ne raditi unaprijed** — dodati samo ako se na testu pokaže da gola detekcija noću nije dovoljna. Ako se doda, primijeniti na kopiju framea koja ide u model, a prikazivati originalni frame.

5. **Realna očekivanja:** Noćna detekcija će biti slabija od dnevne i to je očekivano — za obranu projekta dovoljno je da sustav pouzdano hvata bliža, dobro osvijetljena vozila. Ne trošiti pretjerano vrijeme na savršenu detekciju vozila u dubokoj pozadini.

---

## config.py — parametri koje treba izložiti
```python
import torch

VIDEO_PATH = "video/gradska_voznja_dan.mp4"   # promijeniti na noćni za drugi demo
MODEL_PATH = "yolo11s.pt"                       # s = small; probati i yolo11m.pt
DEVICE = 0 if torch.cuda.is_available() else "cpu"   # 0 = RTX 3050
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck
IMG_SIZE = 640
TRACKER = "bytetrack.yaml"

# --- Profili dan/noć ---
PROFILE = "day"   # "day" ili "night"

PROFILES = {
    "day": {
        "CONF_THRESHOLD": 0.40,
        "MIN_BOX_AREA": 1500,
        "LINE_Y_RELATIVE": 0.70,
        "USE_CLAHE": False,
    },
    "night": {
        "CONF_THRESHOLD": 0.28,   # niži prag — slabije vidljiva vozila
        "MIN_BOX_AREA": 2000,     # agresivnije filtriranje refleksija
        "LINE_Y_RELATIVE": 0.70,
        "USE_CLAHE": False,       # uključiti samo ako detekcija noću podbaci
    },
}
CFG = PROFILES[PROFILE]

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    2: (0, 255, 0),      # zelena
    3: (0, 255, 255),    # žuta
    5: (255, 0, 0),      # plava
    7: (0, 165, 255),    # narančasta
}
```

---

## Pseudokod glavne petlje (main.py)
```
ucitaj model (YOLO)
otvori video (VideoCapture)
init brojaci (total, by_type, set izbrojanih ID-eva)
init dict prethodnih pozicija centroida {id: (x, y)}

dok ima frameova:
    procitaj frame
    results = model.track(frame, classes=VEHICLE_CLASSES, conf=..., persist=True, tracker=...)

    za svaku detekciju u results:
        izvuci box, track_id, class_id, conf
        ako je box_area < MIN_BOX_AREA: preskoci
        izracunaj centroid (donji-srednji rub)

        nacrtaj box (boja po class_id) + label

        ako track_id u prethodnim pozicijama:
            ako je centroid presao liniju (usporedi sa prev pozicijom):
                ako track_id NIJE u izbrojanima:
                    total += 1
                    by_type[class_id] += 1
                    dodaj track_id u izbrojane
        azuriraj prethodnu poziciju[track_id] = centroid

    nacrtaj liniju brojanja
    nacrtaj overlay panel (total, by_type, current_in_frame)
    prikazi frame (imshow)
    ako tipka == 'q': break

ispisi finalnu statistiku
oslobodi resurse
```

---

## Kriteriji uspjeha (definition of done)
- [ ] **Inference-only:** rješenje koristi predtrenirani model bez ikakvog treniranja/fine-tuninga; profesorovi videi nisu korišteni kao trening podaci
- [ ] Video se pokreće i prikazuje uživo bez rušenja
- [ ] Vozila imaju bounding box s ispravnom bojom po tipu i stabilnim ID-em
- [ ] Linija brojanja je vidljiva i podesiva preko `config.py`
- [ ] Brojač raste kad vozilo prijeđe liniju i **ne broji isto vozilo dvaput**
- [ ] Overlay prikazuje ukupan broj, broj po tipu i broj vozila u kadru
- [ ] Radi u (približno) stvarnom vremenu na zadanom videu (cilj ≥15 FPS na RTX 3050; ako ne, dokumentirati skaliranje)
- [ ] Inferencija ide na GPU (`torch.cuda.is_available()` vraća `True`, `device=0`)
- [ ] Rješenje testirano na **oba videa** (dnevni i noćni) uz odgovarajući profil
- [ ] README objašnjava kako pokrenuti, kako prebaciti dan/noć profil i kako se interpretiraju rezultati (uz napomenu o pokretnoj kameri)

---

## Napomene za agenta
- **INFERENCE-ONLY:** Nema treniranja ni fine-tuninga. Ne pisati `model.train()`, ne stvarati dataset, ne anotirati frameove. Samo predtrenirani model → inference na videu. Profesorovi videi su isključivo za demo.
- Prvo napraviti **minimalni radni demo** (detekcija + tracking + crtanje boxova) **na dnevnom videu**, provjeriti da radi i da koristi GPU, pa tek onda dodavati brojanje i statistiku. Inkrementalno.
- Nakon što dnevni radi, **testirati na noćnom** i podesiti `night` profil (niži conf, eventualno CLAHE).
- Provjeriti GPU na startu (`torch.cuda.is_available()`); ako je `False`, problem je u PyTorch instalaciji (krivi build bez CUDA).
- Liniju brojanja **ostaviti lako podesivom** — njena pozicija se mora eksperimentalno namjestiti na svakom videu, ne može se pogoditi unaprijed.
- Ne komplicirati s procjenom brzine — nije u opsegu (kalibracija je nepouzdana kod pokretne kamere).
- CLAHE/gamma korekciju za noć dodati **samo ako gola detekcija podbaci** — ne unaprijed.
- Kod komentirati na hrvatskom ili engleskom dosljedno; cilj je da student može objasniti svaku liniju na obrani projekta.
- Ako `model.track()` s `bytetrack.yaml` ne radi out-of-the-box, fallback na default tracker (`botsort.yaml`) ili samo `model.track(persist=True, device=0)`.
