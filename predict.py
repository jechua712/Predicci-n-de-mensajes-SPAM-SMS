"""
predict.py
----------
Carga el modelo guardado y clasifica mensajes SMS.

Uso (con archivo .txt, un mensaje por línea):
    python predict.py mensajes.txt

Uso (sin archivo, usa mensajes de ejemplo):
    python predict.py
"""

import sys, re, os
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Descargar NLTK data si falta ──────────────────────────────────────────────
for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
stemmer    = PorterStemmer()

# ── Preprocesamiento (idéntico al de entrenamiento) ───────────────────────────
def preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^a-z\s$!]", "", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ── Cargar modelo ─────────────────────────────────────────────────────────────
MODEL_PATH = "spam_model.joblib"

if not os.path.exists(MODEL_PATH):
    print(f"❌ No se encontró el modelo en '{MODEL_PATH}'.")
    print("   Ejecuta primero: python train_model.py")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
print(f"Modelo cargado desde: {os.path.abspath(MODEL_PATH)}\n")

# ── Obtener mensajes ──────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    # Leer desde archivo .txt (un mensaje por línea)
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Archivo no encontrado: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        messages = [line.strip() for line in f if line.strip()]
    print(f"Leyendo {len(messages)} mensaje(s) desde '{filepath}'...\n")
else:
    # Mensajes de ejemplo hardcodeados
    messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
        "Hey, are we still meeting up for lunch today?",
        "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
        "Reminder: Your appointment is scheduled for tomorrow at 10am.",
        "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
    ]
    print("No se indicó archivo. Usando mensajes de ejemplo.\n")

# ── Preprocesar y predecir ────────────────────────────────────────────────────
processed = [preprocess(msg) for msg in messages]
predictions  = model.predict(processed)
probabilities = model.predict_proba(processed)

# ── Mostrar resultados ────────────────────────────────────────────────────────
separator = "─" * 60
print(separator)
for i, msg in enumerate(messages):
    label        = "🚨 SPAM" if predictions[i] == 1 else "HAM (no spam)"
    prob_spam    = probabilities[i][1]
    prob_ham     = probabilities[i][0]

    # Truncar mensajes largos para la pantalla
    display_msg = msg if len(msg) <= 80 else msg[:77] + "..."

    print(f"Mensaje : {display_msg}")
    print(f"Resultado: {label}")
    print(f"Probabilidad spam: {prob_spam:.1%}  |  ham: {prob_ham:.1%}")
    print(separator)
