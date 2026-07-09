# BMW Car Sales Prediction - SAUSAU projekat

Ovaj projekat resava zadatak binarne klasifikacije uspeha prodaje BMW automobila.
Cilj je da se na osnovu karakteristika automobila i trzista predvidi klasa
`Sales_Classification`, odnosno da li prodaja pripada klasi `High` ili `Low`.

Projekat koristi klasicne algoritme masinskog ucenja iz gradiva predmeta:
logisticku regresiju, KNN, Gradient Boosting, stablo odlucivanja i Random Forest.
Ne koriste se neuronske mreze.

## Dataset

Polazni dataset se nalazi u folderu `data/raw/` i sadrzi podatke o prodaji BMW
automobila. Koriste se sledeci atributi:

- kategorijski atributi: `Model`, `Region`, `Color`, `Fuel_Type`, `Transmission`
- numericki atributi: `Year`, `Engine_Size_L`, `Mileage_KM`, `Price_USD`
- ciljna promenljiva: `Sales_Classification`

Atribut `Sales_Volume` se ne koristi u glavnom prediktivnom modelu zato sto
direktno odredjuje ciljnu promenljivu i predstavlja curenje informacija
(`data leakage`). U projektu postoji posebna analiza koja pokazuje da pravilo
`Sales_Volume >= 7000` dobija 100% tacnost, zbog cega bi njegovo koriscenje
dalo nerealno dobar model.

## Obrada podataka

U fajlu `src/data_preparation.py` rade se osnovni koraci pripreme podataka:

- ucitavanje dataseta
- provera nedostajucih vrednosti
- provera duplikata
- provera raspodele ciljnih klasa
- izdvajanje ulaznih atributa i ciljne promenljive
- analiza anomalija i ekstremnih vrednosti
- analiza odnosa kategorijskih atributa i ciljne klase
- posebna analiza `Sales_Volume` atributa zbog moguceg data leakage problema

Kategorijski atributi se u modelima obradjuju pomocu one-hot enkodiranja, dok se
numericki atributi prosledjuju modelima kao numericke vrednosti. Kod KNN i
logisticke regresije numericki atributi se dodatno skaliraju, jer su ti algoritmi
osetljivi na razlicite opsege vrednosti.

## Podela podataka

Dataset je podeljen na tri dela:

- trening skup - za treniranje modela
- validacioni skup - za izbor najboljeg modela
- test skup - za finalnu proveru izabranog modela

Test skup se koristi tek na kraju, nakon izbora modela na osnovu validacionog
skupa. Na taj nacin se izbegava da se model indirektno prilagodjava test podacima.

## Modeli

Modeli su definisani u fajlu `src/modeling.py`. Isprobani su:

- Logistic Regression
- KNN
- Gradient Boosting
- Decision Tree
- Random Forest

Za proveru stabilnosti modela koristi se unakrsna validacija sa metrikom
`f1_macro`. Najbolji model se bira na osnovu rezultata na validacionom skupu, a
finalni rezultat se zatim proverava na test skupu.

## Evaluacija

Evaluacija modela se nalazi u fajlu `src/evaluation.py`. Racunaju se sledece
metrike:

- accuracy
- precision za klase `High` i `Low`
- recall za klase `High` i `Low`
- f1-score za klase `High` i `Low`
- macro average
- weighted average
- ROC AUC
- matrica konfuzije

Finalni izvestaj se cuva u:

```text
results/metrics/final_classification_report.txt
```

Trenutni finalni model je `Decision Tree`. Na test skupu daje sledece rezultate:

- accuracy: 0.5471
- High f1-score: 0.3531
- Low f1-score: 0.6516
- macro f1-score: 0.5023
- ROC AUC: 0.5091

## Najbitniji atributi

U projektu se racuna znacaj atributa i porede se rezultati modela kada se koriste:

- svi atributi
- samo najbitniji atributi

Najbitniji atributi se ne biraju rucno fiksiranjem tacnog broja, vec se izdvajaju
na osnovu vrednosti znacaja atributa. Rezultati poredjenja nalaze se u:

```text
results/metrics/all_vs_important_features_comparison.csv
results/metrics/selected_important_features.csv
```

## Rezultati i grafici

Svi tabelarni rezultati nalaze se u folderu:

```text
results/metrics/
```

Najvazniji fajlovi su:

- `model_comparison.csv` - poredjenje modela
- `validation_model_comparison.csv` - rezultati na validacionom skupu
- `test_model_comparison.csv` - rezultati na test skupu
- `final_model_metrics.csv` - metrike finalnog modela
- `final_classification_report.txt` - pregled metrika po klasama i zbirno
- `feature_importance.csv` - znacaj atributa
- `correlation_matrix.csv` - korelacije numerickih atributa
- `sales_volume_leakage_analysis.csv` - analiza data leakage problema

Grafici se nalaze u folderu:

```text
results/figures/
```

Tu se nalaze raspodela klasa, histogrami, boxplot grafici, korelaciona matrica,
feature importance grafik, matrica konfuzije i grafici odnosa kategorijskih
atributa sa ciljnom klasom.

## Deployment

Finalni istrenirani model se eksportuje pomocu biblioteke `joblib` u fajl:

```text
models/bmw_sales_model.joblib
```

Model se moze koristiti na tri nacina:

- direktno kroz terminal pomocu `src/predict.py`
- kroz FastAPI servis u `app/api.py`
- kroz Streamlit UI u `app/ui.py`

## Pokretanje projekta

Instalacija biblioteka:

```bash
pip install -r requirements.txt
```

Treniranje i evaluacija:

```bash
python src/main.py
```

Predikcija iz terminala:

```bash
python src/predict.py
```

FastAPI deployment:

```bash
uvicorn app.api:app --reload
```

Streamlit UI:

```bash
streamlit run app/ui.py
```

## Struktura projekta

```text
app/                 API i Streamlit UI za koriscenje modela
data/                polazni dataset
models/              eksportovani istrenirani model
results/figures/     sacuvani grafici
results/metrics/     sacuvane metrike i tabele
src/                 glavni kod projekta
```

## Dokumentacija

Detaljnija projektna dokumentacija se nalazi u Word dokumentu:

```text
sausau_projektna_dokumentacija.docx
```

Dokumentacija opisuje problem, obradu podataka, treniranje modela, evaluaciju,
izbor atributa, deployment i tumacenje dobijenih rezultata.
