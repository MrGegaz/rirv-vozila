# Istraživanje tehnologija za rješenje vlastitog zadatka

**Predmet:** Računarski i robotski vid
**Zadatak:** Detekcija i analiza vozila na videu (gradska vožnja)

---

## Opis zadatka

Izradit ću Python aplikaciju koja obrađuje video gradske vožnje i u stvarnom vremenu detektira vozila, prati ih kroz frameove (tracking), broji i razvrstava po tipu (automobil, kamion, autobus, motocikl). Video se obrađuje uživo, bez treniranja modela — koristim gotov predtrenirani model. Koristim isključivo open-source tehnologije.

---

## 1. Tehnologije

- **Python** — programski jezik.
- **Ultralytics YOLO11** — detekcija vozila pomoću predtreniranog modela (COCO skup, 80 klasa). https://docs.ultralytics.com/models/yolo11
- **ByteTrack** — praćenje vozila kroz frameove, dolazi ugrađen u Ultralytics paket. https://docs.ultralytics.com/modes/track
- **OpenCV** — učitavanje videa, iscrtavanje okvira i prikaz uživo. https://docs.opencv.org/
- **PyTorch + NumPy** — pozadinski okvir na kojem radi YOLO (uz NVIDIA GPU) i rad s nizovima.

Odabrao sam YOLO umjesto starijih OpenCV kaskadnih klasifikatora jer daje bolju točnost u složenim scenama poput gradskog prometa. YOLO je dostupan pod AGPL-3.0 open-source licencom, pa odgovara uvjetu zadatka.

---

## 2. Algoritmi i pristupi

**YOLO (You Only Look Once)** — model za detekciju objekata razvijen 2015. Osnovna ideja je da cijelu sliku obradi u jednom prolazu i odjednom predvidi granične okvire i klase objekata, zbog čega je vrlo brz i prikladan za rad u stvarnom vremenu. Koristim YOLO11, izdan 2024., koji dobro radi detekciju u stvarnom vremenu.

**ByteTrack** — algoritam za praćenje više objekata. Za razliku od starijih trackera, ne odbacuje detekcije niske pouzdanosti nego koristi i njih, što pomaže kod djelomično zaklonjenih vozila (česta pojava u gradskom prometu). Tako svako vozilo dobiva stabilan ID kroz frameove.

---

## 3. Koraci rješenja

1. **Učitavanje videa** — OpenCV-om čitam video frame po frame.
2. **Detekcija vozila** — svaki frame šaljem YOLO11 modelu, ograničeno na klase vozila (automobil, motocikl, autobus, kamion). Radi se o inferenciji (korištenje gotovog modela), ne treniranju.
3. **Praćenje** — koristim `model.track()` s ByteTrackom da svako vozilo dobije jedinstveni ID.
4. **Brojanje** — postavim virtualnu liniju i kad vozilo ju prijeđe, povećam brojač (svako vozilo se broji samo jednom preko njegovog ID-a).
5. **Klasifikacija po tipu** — vodim zasebne brojače po tipu vozila i razlikujem ih bojom okvira.
6. **Prikaz uživo** — iscrtavam okvire, ID-eve i statistiku (ukupno, po tipu, trenutno u kadru).

**Napomena:** Kamera je u pokretu (snimka iz vozila), pa brojač predstavlja broj vozila koja su prošla kroz scenu, a ne klasičan prometni protok. Jedan video je dnevni, drugi noćni — kod noćnog koristim niži prag pouzdanosti zbog slabijeg svjetla.

---

## Izvori

- Ultralytics YOLO11 — https://docs.ultralytics.com/models/yolo11
- Ultralytics Track — https://docs.ultralytics.com/modes/track
- OpenCV — https://docs.opencv.org/
- ByteTrack — https://trackers.roboflow.com/latest/trackers/bytetrack/
