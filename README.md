# BMW Car Sales Prediction - SAUSAU projekat

Projekat koristi klasicne algoritme masinskog ucenja iz gradiva predmeta:
logisticku regresiju, KNN, Gradient Boosting, stablo odlucivanja i Random Forest.

Ne koriste se neuronske mreze.

## Pokretanje

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

## Vazna napomena

`Sales_Volume` se ne koristi u glavnom prediktivnom modelu zato sto direktno
odredjuje ciljnu promenljivu `Sales_Classification`. Projekat ipak sadrzi
posebnu leakage analizu koja pokazuje da pravilo `Sales_Volume >= 7000`
dobija 100% tacnost.
