# Detekcija i praćenje vozila (Računarski i robotski vid)

Live demo sustav koji nad videima **gradske vožnje iz perspektive vozila u pokretu**
(dashcam) u stvarnom vremenu radi:

1. **Detekciju vozila** (auto, kamion, autobus, motocikl)
2. **Praćenje (tracking)** istih vozila kroz frameove uz stabilan ID
3. **Klasifikaciju** tipa vozila i prikaz broja vozila trenutno u kadru po tipu
4. **Prepoznavanje boje** vozila (HSV analiza karoserije) i ispis u label — danju
5. **Trajektorije (tragove kretanja)** svakog praćenog vozila
6. **Statistiku** na kraju (broj vozila po tipu, max u kadru) + CSV log

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
| `DETECT_COLOR`   | True | False | Boja karoserije; noću je vozilo u mraku pa je nepouzdano |

---

## Prepoznavanje boje vozila

Za svako vozilo procjenjuje se dominantna boja karoserije i ispisuje u zasebnom
retku labela (ispod `ID / tip / pouzdanost`). Klasičan CV, bez treniranja:

- Uzme se **središnji pojas karoserije** (srednjih 50% širine, 40–75% visine) da
  se izbjegnu cesta/pozadina, stakla i kotači.
- Uzorak se pretvori u **HSV** (odvaja ton boje `H` od osvjetljenja `V`); uzima se
  **medijan** (otpornost na pojedine svijetle piksele / refleksije).
- Niska zasićenost `S` → akromatska boja (bijela / siva / crna po svjetlini `V`);
  inače kromatska boja po tonu `H` (crvena, žuta, zelena, plava).
- Boja se **glasa kroz zadnjih `COLOR_VOTE_LENGTH` frameova** po track ID-u da
  label ne treperi.

Uključuje se po profilu (`DETECT_COLOR`): **danju** je pouzdano, **noću isključeno**
jer je karoserija u mraku (vide se samo svjetla), pa bi rezultat bio nasumičan.

### Podešavanje pragova boje

Granica bijela/siva/crna inherentno je mutna, pa su pragovi u `config.py`:

- `COLOR_SAT_MIN` — granica akromatsko/kromatsko (zasićenost `S`).
- `COLOR_VAL_WHITE` — iznad ovog (uz nisku `S`) je bijela; spustiti ako se bijeli
  auti zovu "gray", podići ako se srebrni zovu "white".
- `COLOR_VAL_BLACK` — ispod ovog je crna.

Za precizno podešavanje postaviti `COLOR_DEBUG = True`: uz label se ispiše izmjereno
`H,S,V` pa se na konkretnom vozilu očita prava vrijednost i namjesti prag.

---

## Zona vlastitog vozila (hauba)

Kod dashcama dio vlastitog auta (hauba/armatura) viri na dnu kadra i zna se
detektirati kao vozilo (osobito noću). Zato se detekcije čiji je vertikalni centar
**ispod** `IGNORE_BOTTOM_REL` (udio visine kadra) ignoriraju. Tanka siva linija u
kadru pokazuje granicu zone radi lakšeg podešavanja.

- `IGNORE_BOTTOM_REL = 0.85` (default) — sve ispod 85% visine se reže.
- Smanji broj (npr. 0.80) ako hauba i dalje ulazi u detekciju; povećaj (ili `1.0`)
  ako reže stvarna bliska vozila.

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
├── analytics.py       # trajektorije (TrailTracker) + boja (ColorVoter) + statistika/CSV (StatsCollector)
├── config.py          # svi parametri + profili day/night
├── requirements.txt
├── README.md
├── .gitignore
└── video/             # videi (nisu u gitu)
```

## COCO klase vozila
`2` = car, `3` = motorcycle, `5` = bus, `7` = truck. Detekcija je ograničena na njih.
