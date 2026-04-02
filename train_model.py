"""
train_model.py
--------------
Descarga el dataset, entrena el modelo de detección de spam y lo guarda en disco.
Ejecutar una sola vez (o cuando quieras reentrenar).

Uso:
    python train_model.py
"""

import requests, zipfile, io, os, re
import pandas as pd
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# ── 1. Descargar NLTK data ────────────────────────────────────────────────────
for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
stemmer    = PorterStemmer()

# ── 2. Descargar y cargar dataset ─────────────────────────────────────────────
print("Descargando dataset...")
url      = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = requests.get(url)
assert response.status_code == 200, "No se pudo descargar el dataset."

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")

df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t", header=None, names=["label", "message"]
)
df = df.drop_duplicates()
print(f"Dataset cargado: {len(df)} mensajes  |  spam={df['label'].value_counts()['spam']}  ham={df['label'].value_counts()['ham']}")

# ── 3. Preprocesamiento ───────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^a-z\s$!]", "", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

print("Preprocesando mensajes...")
df["message"] = df["message"].apply(preprocess)
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

# ── 4. Pipeline + Grid Search ─────────────────────────────────────────────────
pipeline = Pipeline([
    ("vectorizer", CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])

param_grid = {"classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]}

print("Buscando mejores hiperparámetros (puede tardar unos segundos)...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(df["message"], y)

best_model = grid_search.best_estimator_
print(f"Mejor alpha: {grid_search.best_params_['classifier__alpha']}")

# ── 5. Evaluación rápida (sobre todo el set, solo referencial) ────────────────
preds = best_model.predict(df["message"])
print("\n── Reporte de clasificación ──")
print(classification_report(y, preds, target_names=["ham", "spam"]))

# ── 6. Guardar modelo ─────────────────────────────────────────────────────────
MODEL_PATH = "spam_model.joblib"
joblib.dump(best_model, MODEL_PATH)
print(f"\n✅ Modelo guardado en: {os.path.abspath(MODEL_PATH)}")
