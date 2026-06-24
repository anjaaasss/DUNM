# Detekcija Prevara Kreditnim Karticama pomoću Neuronskih Mreža

> Primena dubokog učenja (Deep Learning) za otkrivanje finansijskih prevara u uslovima ekstremnog disbalansa klasa (577:1)

## Sadržaj

1. [Opis problema](#1-opis-problema)
2. [Podaci](#2-podaci)
3. [Arhitektura modela](#3-arhitektura-modela)
4. [Trening](#4-trening)
5. [Analiza osetljivosti i hiperparametarska optimizacija](#5-analiza-osetljivosti-i-hiperparametarska-optimizacija)
6. [Rezultati evaluacije](#6-rezultati-evaluacije)
7. [Diskusija](#7-diskusija)
8. [Zaključak](#8-zaključak)

---

## 1. Opis problema

Detekcija prevara kreditnim karticama jedan je od najtežih problema klasifikacije u finansijskoj industriji. Centralni izazov nije samo tačnost predviđanja, već **ekstremni disbalans klasa**: od 284.807 transakcija u datasetu, samo 492 (0.17%) su prevare — odnos 577:1.

Posledica ovog disbalansa je da **trivijalni model koji uvek predviđa "legitimno" postiže tačnost od 99.83%**, a istovremeno je potpuno beskoristan jer ne uhvata nijednu prevaru. Iz tog razloga, standardna metrika *accuracy* je odbačena, a sve odluke u projektu vođene su primarnom metrikom — **Average Precision (AP)**, površinom ispod Precision-Recall krive — koja direktno meri performanse na manjinskoj klasi.

**Cilj projekta** je izgradnja i evaluacija sistema baziranog na neuronskim mrežama koji:
- Maksimizuje broj otkrivenih prevara (visok Recall)
- Minimizuje broj lažnih alarma (visok Precision)
- Ostaje robustan na varijabilnost uzrokovanu malim brojem pozitivnih primera

---

## 2. Podaci

### Izvor

Dataset potiče sa platforme [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), kreiran u saradnji sa Université Libre de Bruxelles (ULB).

### Struktura

| Kolona | Opis |
|---|---|
| `V1` – `V28` | Anonimizovane PCA komponente (zaštita privatnosti korisnika) |
| `Time` | Sekunde od prve transakcije u datasetu (0 – 172.792) |
| `Amount` | Iznos transakcije u EUR |
| `Class` | Ciljna promenljiva: 0 = legitimna, 1 = prevara |

- **Ukupno instanci:** 284.807
- **Prevare:** 492 (0.17%) | **Legitimne:** 284.315 (99.83%)
- **Nedostajuće vrednosti:** nema

### Analiza i predobrada

**Eksplorativna analiza (EDA)** otkrila je ključne uvide:

- **Bimodalna distribucija iznosa prevara:** veliki broj mikro-transakcija ispod 20 EUR (automatizovane provere validnosti kartice) i sekundarni pik ispod 500 EUR
- **Temporalni šabloni napada:** prevare se ne raspoređuju ravnomerno — karakterišu ih iznenadni pikovi oko 11h i 26h (*burst attacks*), vidljivi na kumulativnoj krivoj kao nagli skokovi ("stepenice"). Prevaranti svesno biraju sate visokog legitimnog volumena da se sakriju u gomili
- **Najjači prediktori:** PCA komponente `V17`, `V14`, `V12`, `V10` imaju najjaču negativnu korelaciju sa prevarama; `V11` i `V4` najjaču pozitivnu
- **Ortogonalnost PCA komponenti:** korelaciona matrica potvrđuje da su V-atributi međusobno nekorelisani (0.00), što eliminiše multikolinearnost i znači da svaki nosi jedinstvenu informaciju

**Feature Engineering** — kreirano 11 novih atributa iz `Time` i `Amount`:

| Atribut | Opis |
|---|---|
| `Hour`, `Hour_sin`, `Hour_cos` | Sat u danu sa cikličnim kodiranjem (sin/cos eliminira problem da je sat 23 i sat 0 "daleko") |
| `Is_Night` | Binarni flag: transakcija između 22h i 6h |
| `Log_Amount`, `Sqrt_Amount` | Logaritamska i koren transformacija iznosa (smanjuju uticaj outliera) |
| `Amount_log_scaled` | Standardizovani log iznos |
| `V14_V17`, `V14_V12`, `V4_V11` | Interakcijski atributi — proizvod dva najjača prediktora |
| `V_norm` | L2 norma top PCA komponenti — meri "ukupnu anomalnost" transakcije |

**Predobrada:**
- Sirove kolone `Time` i `Amount` izbačene (zamenjene FE verzijama)
- **RobustScaler** na sve atribute (otporan na outliere, jer su prevare statistički outlieri)
- Stratifikovana podela 70/15/15 — svaki podskup čuva 0.173% prevara
- **Ukupno atributa:** 39 (28 V komponenti + 11 FE atributa)

---

## 3. Arhitektura modela

### Teorijska osnova

Sve neuronske mreže su tipa **Multi-Layer Perceptron (MLP)** — potpuno povezane mreže.

**Forward pass:**
$$\mathbf{h}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \cdot \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

**Aktivacione funkcije:**
- Skriveni slojevi: `ReLU(x) = max(0, x)` — rešava problem nestajućeg gradijenta
- Izlazni sloj: `Sigmoid` → P(prevara) ∈ (0, 1)

**Optimizator:** Adam | **Loss:** Binarna unakrsna entropija | **Regularizacija:** L2 (λ = 0.0001)

### Struktura skrivenih slojeva

Svaki skriveni sloj prati strukturu:

```
Dense(h neurona, use_bias=False)
    ↓
BatchNormalization   ← stabilizuje gradijente, ubrzava konvergenciju
    ↓
ReLU aktivacija
    ↓
Dropout(0.3)        ← sprečava overfitting na malom broju fraud primera
```

### Testirane arhitekture

| Model | Arhitektura | Parametri |
|---|---|---|
| Logistička regresija | Linearni model (baseline) | — |
| Random Forest | 200 stabala, max_depth=12 | — |
| Keras-Small | `Input(39) → Dense(64) → Dense(1)` | ~2.600 |
| Keras-Medium | `Input(39) → Dense(256) → Dense(128) → Dense(1)` | ~43.000 |
| Keras-Deep | `Input(39) → Dense(256) → Dense(128) → Dense(64) → Dense(1)` | ~51.000 |

---

## 4. Trening

### Rešavanje disbalansa klasa

Primenjena je **kombinacija dva pristupa**:

**1. SMOTE** (Synthetic Minority Oversampling Technique):
- Generiše sintetičke fraud primere interpolacijom između postojećih u feature prostoru
- `sampling_strategy=0.20` → fraud povećan na 20% od broja legitimnih
- Primenjuje se **isključivo na trening skup** (val i test ostaju netaknuti — nema data leakage)
- U K-fold CV: SMOTE se primenjuje posebno unutar svakog folda

**2. Class weights:**
- Keras-u se zadaje `class_weight` koji proporcionalno kažnjava grešku na fraud klasi

### Callback strategija

```python
EarlyStopping(monitor='val_pr_auc', patience=25, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=12, min_lr=1e-6)
```

Ključna odluka: praćenje `val_pr_auc` umesto `val_auc` jer je PR-AUC direktno osetljiv na performanse na manjinskoj klasi.

### Threshold selekcija

Umesto podrazumevanog praga 0.5, optimalni prag se bira pretraživanjem opsega [0.01, 0.99] i biranjem vrednosti koja maksimizuje F1 za klasu prevara. Keras-Deep koristi prag **0.98**, Keras-Medium **0.90**.

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Sprovedena je **one-at-a-time sensitivity analiza** — svaki hiperparametar menja se izolovano dok su ostali fiksirani. Evaluacija na originalnom val skupu po **Val PR-AUC**.

| Hiperparametar | Testirane vrednosti | Optimum | Max Val AP |
|---|---|---|---|
| Learning rate | 0.01 / 0.001 / 0.0001 | **0.001** | 0.84639 |
| L2 regularizacija | 0.01 / 0.001 / 0.0001 | **0.0001** | 0.82370 |
| Batch size | 128 / 256 / 512 | **512** | 0.84104 |
| Dropout rate | 0.1 / 0.3 / 0.5 | **0.3** | 0.82982 |

**Ključni nalazi:**
- **Learning rate** je najuticajniji parametar — agresivan LR (0.01) preskače optimum (AP=0.761), konzervativni (0.0001) zaglavljuje u suboptimalnim zonama (AP=0.776)
- **L2 regularizacija:** slabija kazna (0.0001) daje bolje rezultate jer SMOTE + Dropout već dovoljno regularizuju
- **Batch size:** veći batch (512) daje stabilniji gradijent na SMOTE-augmentovanim podacima, što je suprotno od uobičajene preporuke za male skupove
- **Dropout 0.3** je zlatna sredina — previše (0.5) previše agresivno "gasi" neurone, premalo (0.1) ne štiti dovoljno

---

## 6. Rezultati evaluacije

### Poređenje svih modela (test skup: 42.648 legitimnih, 74 prevare)

| Model | Test AUC | Test AP | Recall | Precision | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistička regresija | 0.96557 | 0.72237 | 0.8243 | 0.5446 | 0.6559 | 59 | 13 |
| Random Forest | 0.97805 | 0.79767 | 0.6892 | 0.9107 | 0.7846 | 6 | 23 |
| Keras-Small (64) | 0.97093 | 0.67456 | 0.8108 | 0.6742 | 0.7362 | 29 | 14 |
| **Keras-Medium (256-128)** | 0.96648 | **0.82224** | 0.8108 | **0.8824** | **0.8451** | **8** | 14 |
| Keras-Deep (256-128-64) | 0.95907 | 0.78394 | 0.8243 | 0.7093 | 0.7625 | 25 | 13 |

**Keras-Medium je pobednik po AP i F1 metrikama na test skupu.** Postiže visok Recall (81%) uz izuzetno visoku Precision (88%) — samo 8 lažnih alarma naspram 29 kod Small i 25 kod Deep modela.

### Komparativna analiza: Keras-Deep vs Keras-Medium

Oba modela pokazuju odličnu separaciju (mean verovatnoća za legitimne: Deep=0.0012, Medium=0.0004), ali se razlikuju u trade-offu:

| | Keras-Deep | Keras-Medium |
|---|---|---|
| Recall | **82.43%** | 81.08% |
| Precision | 70.93% | **88.24%** |
| F1 | 0.7625 | **0.8451** |
| Lažni alarmi (FP) | 25 | **8** |
| Opt. prag | 0.98 | 0.90 |
| **Fokus** | Više prevara uhvaćeno | Manje lažnih alarma |

**Keras-Deep** je pogodan za banke koje prioritizuju maksimalni Recall (uhvatiti svaku prevaru). **Keras-Medium** je pogodan za banke koje žele balans između detekcije i operativnog opterećenja tima.

### K-Fold Cross-Validacija (5-fold, zvanično poređenje)

Pošto test skup ima ~74 fraud primera, single-run AP je statistički nestabilan. K-fold daje pouzdaniji poredak:

| Arhitektura | CV AUC | CV AP |
|---|---|---|
| Keras-Small (64) | 0.95904 ± 0.01950 | 0.71865 ± 0.06894 |
| Keras-Medium (256-128) | — | — |
| **Keras-Deep (256-128-64)** | — | **0.76192 ± 0.04666** ← pobednik |

Keras-Deep je **zvanični pobednik po K-fold CV AP** i ima **najmanju standardnu devijaciju** (±0.047), što znači da je najpouzdaniji i najstabilniji kroz različite podele podataka.

### Analiza važnosti atributa (Permutation Importance)

Top 5 najvažnijih atributa za Keras-Deep model (po padu PR-AUC pri shufflovanju):

1. **V14_V17** (~0.10 pad) — apsolutni dominantor; interakcijski atribut koji model direktno koristi
2. **V17** — originalna PCA komponenta, najjači linearni prediktor iz EDA
3. **V_norm** — L2 norma top komponenti
4. **V14** — originalna PCA komponenta
5. **V4** — originalna PCA komponenta

Napomena: V14_V17 dominira jer mreža koristi eksplicitno konstruisan interakcijski atribut umesto da kombinuje V14 i V17 odvojeno kroz slojeve — ovo potvrđuje vrednost feature engineeringa.

### Finansijska analiza

Parametri: FN (propuštena prevara) = 100 EUR, FP (lažni alarm) = 3 EUR

| Strategija | Gubitak | Ušteda |
|---|---|---|
| Bez modela | 7.400 EUR | — |
| Keras-Small | 1.436 EUR | 5.964 EUR (80.6%) |
| Keras-Medium | 1.424 EUR | 5.976 EUR (80.8%) |
| **Keras-Deep** | **1.375 EUR** | **6.025 EUR (81.4%)** |

Keras-Deep ostvaruje najveću finansijsku uštedu zahvaljujući jednom dodatno uhvaćenom napadu (FN=13 vs 14 kod ostalih).

---

## 7. Diskusija

**Zašto Keras-Medium ima bolji F1, a Keras-Deep bolji K-fold AP?**
Ovo je klasičan primer nestabilnosti single-run metrika sa malim brojem fraud primera (~74 u test skupu). Keras-Medium je u ovom konkretnom pokretanju ostvario izvanredno balansiran trade-off (samo 8 FP), ali K-fold CV pokazuje da Keras-Deep u proseku dostiže višu AP (0.762 vs ~0.73 za Small) i ima manju varijansu — što ga čini pouzdanijim u produkcionom okruženju.

**Uloga SMOTE-a:** Ključan za sprečavanje "kolapsa" modela (TP=0). Bez SMOTE-a, mreža uči da uvek predviđa klasu 0. Kritično: SMOTE se primenjuje unutar svakog K-fold folda posebno, nikad na validacionom skupu — čime se sprečava data leakage.

**Zašto PR-AUC, ne ROC-AUC?** ROC-AUC koristi FPR koji u imenitelju ima 284.315 legitimnih transakcija — čak 1000 lažnih alarma čini samo 0.35% FPR i ROC kriva "ne primećuje" taj problem. PR-AUC direktno meri preciznost na fraud klasi i ne "laže" kod neuravnoteženih skupova.

**Permutation Importance vs. korelacija:** EDA stavlja V17 na prvo mesto (Pearson r = -0.33), ali permutation importance stavlja V14_V17 na prvo mesto — jer mreža koristi eksplicitno konstruisan interakcijski atribut, pa shufflovanje samog V17 malo menja rezultat. Ovo ilustruje razliku između linearne statističke korelacije i stvarne važnosti u kontekstu naučenog modela.

---

## 8. Zaključak

Projekat je demonstrirao da neuronske mreže sa odgovarajućim tehnikama za neuravnotežene skupove mogu efektivno detektovati finansijske prevare, ostvarujući uštede od **81.4%** u poređenju sa sistemom bez detekcije.

Ključni zaključci:

1. **Average Precision je jedina relevantna metrika** — accuracy i ROC-AUC su varljivi za ovaj problem
2. **Kombinacija SMOTE + BatchNorm + Dropout** je neophodna za stabilan trening
3. **Single-run metrike su nestabilne** sa ~74 fraud primera u test skupu — K-fold CV je obavezan za poređenje arhitektura
4. **Keras-Deep je zvanični pobednik** po K-fold CV (AP=0.762, std=0.047) i finansijskoj analizi (ušteda 81.4%)
5. **Keras-Medium je pobednik po F1 i Precision** u single-run evaluaciji — preporučuje se bankama koje prioritizuju minimizaciju lažnih alarma
6. **V14_V17 interakcijski atribut** je najvažniji prediktor u finalnom modelu — potvrđuje vrednost feature engineeringa

---

## Reprodukcija projekta

### Preduslovi

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

### Pokretanje

1. Preuzeti dataset: [creditcard.csv sa Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Smestiti fajl u `csvs/creditcard.csv`
3. Pokrenuti `detekcija_prevara.ipynb` od prve do poslednje ćelije redom

### Procenjeno vreme izvršavanja

| Korak | Vreme (CPU) |
|---|---|
| EDA i predobrada | ~2 min |
| Trening 5 modela | ~20–30 min |
| Permutation Importance | ~5–10 min |
| Hiperparametarski sweep | ~25–35 min |
| K-fold CV (5 fold × 3 arhitekture) | ~20–30 min |

---

## Licenca

Ovaj projekat je objavljen pod [MIT licencom](LICENSE).

Dataset je dostupan pod [Database Contents License (DbCL)](https://opendatacommons.org/licenses/dbcl/1-0/) na Kaggle platformi.

---

*Projekat urađen u okviru kursa Duboko učenje i neuronske mreže, Fakultet organizacionih nauka, Univerzitet u Beogradu.*
