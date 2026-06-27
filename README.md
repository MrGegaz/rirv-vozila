# Detekcija i praćenje vozila (Računarski i robotski vid)

Live demo sustav koji nad videima **gradske vožnje iz perspektive vozila u pokretu**
(dashcam) u stvarnom vremenu radi:

1. **Detekciju vozila** (auto, kamion, autobus, motocikl)
2. **Praćenje (tracking)** istih vozila kroz frameove uz stabilan ID
3. **Klasifikaciju** tipa vozila i prikaz broja vozila trenutno u kadru po tipu
4. **Trajektorije (tragove kretanja)** svakog praćenog vozila
5. **Statistiku** na kraju (broj vozila po tipu, max u kadru) + CSV log

Sustav radi na dva videa — **dnevnoj** i **noćnoj** vožnji — uz zasebne profile
parametara.

> **INFERENCE-ONLY:** koristi se isključivo predtrenirani COCO model (`yolo11s.pt`).
> Nema treniranja, fine-tuninga ni stvaranja dataseta. Profesorovi videi služe
> **samo za live demo / evaluaciju**, ne za razvoj ni treniranje.

---

## Tehnologije
- Python 3.10+
- [ultralytics](https://docs.ultralytics.com/) (YOLO11) — detekcija + ugrađeni tracking (ByteTrack)
- OpenCV — čitanje videa, crtanje, prikaz
- NumPy
- PyTorch s CUDA podrškom (GPU ubrzanje)

Hardver razvoja: **NVIDIA RTX 3050 6GB** (CUDA, `device=0`).

---

## Instalacija

```bash
# 1. Virtualno okruženje
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS

# 2. PyTorch s CUDA (za RTX 3050 / noviji driver -> cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Ostale ovisnosti
pip install -r requirements.txt
```

> Ako sustav nema `python3-venv` / `pip` (npr. Debian/Ubuntu): instalirati
> `sudo apt install python3-venv python3-pip`, ili bootstrapati pip preko
> `get-pip.py` u venv napravljen s `python3 -m venv --without-pip .venv`.

**Provjera GPU-a:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
Mora ispisati `CUDA: True`. Ako je `False`, PyTorch je instaliran bez CUDA buildova
— reinstalirati s ispravnim `--index-url`.

Model `yolo11s.pt` se preuzme automatski pri prvom pokretanju.

---

## Pokretanje

1. Staviti video u `video/` i upisati putanju u `config.py` (`VIDEO_PATH`).
2. Odabrati profil u `config.py`:
   ```python
   PROFILE = "day"     # ili "night"
   ```
3. Pokrenuti:
   ```bash
   python main.py
   ```

### Kontrole tijekom reprodukcije

| Tipka   | Akcija                          |
|---------|---------------------------------|
| `q`     | izlaz                           |
| `space` | pauza / nastavak                |
| `-`     | uspori reprodukciju             |
| `+`     | ubrzaj reprodukciju             |

Brzina reprodukcije (0.25x–4x) i FPS prikazani su u overlay panelu.

### Trajektorije i statistika

- Iza svakog vozila crta se **trag kretanja** (zadnjih `TRAIL_LENGTH` pozicija),
  obojen po tipu vozila. Uključuje se/isključuje preko `SHOW_TRAILS` u `config.py`.
- Na kraju (ili na `q`) u konzolu se ispiše **sažetak**: broj vozila po tipu i
  najviše vozila istovremeno u kadru.
- Ako je `SAVE_CSV = True`, sprema se per-frame log u `CSV_PATH`
  (`izlaz_statistika.csv`) sa stupcima: `frame, time_s, total, car, motorcycle, bus, truck`.

> **Napomena (pokretna kamera):** broj po tipu temelji se na track ID-evima.
> ByteTrack povremeno izgubi i ponovno dodijeli ID (okluzije, trešnja), pa je to
> **gruba procjena**, ne točan broj fizičkih vozila.

---

## Dan / noć profil

Profili se prebacuju **jednom varijablom** `PROFILE` u `config.py`. Razlikuju se po:

| Parametar        | day  | night | Zašto                                            |
|------------------|------|-------|--------------------------------------------------|
| `CONF_THRESHOLD` | 0.40 | 0.28  | Noću su vozila slabije vidljiva (samo svjetla)   |
| `MIN_BOX_AREA`   | 1500 | 2000  | Noću agresivnije filtriranje refleksija/blještanja |
| `USE_CLAHE`      | False| False | Pojačanje kontrasta — uključiti SAMO ako noćna detekcija podbaci |

---

## Interpretacija rezultata (pokretna kamera)

Kamera je **u vozilu u pokretu**, nije fiksna nadzorna kamera. Zato:

- Cijela scena se pomiče; parkirana vozila "putuju" kroz kadar.
- Prikazani broj je **broj vozila trenutno vidljivih u kadru** (po tipu), a ne
  prometni protok.
- Tracker (ByteTrack) povremeno izgubi i ponovno dodijeli ID zbog trešnje i okluzija
  — to je očekivano i prihvatljivo za demo.
- Noćna detekcija je slabija od dnevne; sustav pouzdano hvata bliža, dobro
  osvijetljena vozila.

---

## Struktura projekta

```
RiRV_Vozila/
├── main.py            # entry point, live demo petlja (detekcija + tracking + prikaz)
├── analytics.py       # trajektorije (TrailTracker) + statistika/CSV (StatsCollector)
├── config.py          # svi parametri + profili day/night
├── requirements.txt
├── README.md
├── .gitignore
└── video/             # videi (nisu u gitu)
```

## COCO klase vozila
`2` = car, `3` = motorcycle, `5` = bus, `7` = truck. Detekcija je ograničena na njih.
