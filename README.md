# Deteccion de Spam en SMS — Clasificador de Machine Learning

Clasificador de spam para mensajes SMS listo para produccion, construido con Python y scikit-learn, entrenado sobre el dataset SMS Spam Collection de la UCI. Desarrollado como parte del camino de aprendizaje del rol **AI Red Teamer** en [Hack The Box](https://www.hackthebox.com).

---

## Contexto

Este proyecto fue construido mientras se trabajaba el rol AI Red Teamer en Hack The Box, que cubre tecnicas de seguridad ofensiva aplicadas a sistemas de machine learning — incluyendo evasion de modelos, entradas adversariales y como engañar clasificadores. Construir un detector de spam desde cero da la comprension base de como funcionan estos modelos antes de aprender a romperlos.

---

## Que hace

- Descarga y preprocesa el dataset SMS Spam Collection (5,574 mensajes)
- Limpia y tokeniza el texto, elimina stopwords y aplica stemming
- Extrae features usando un modelo bag-of-words con bigramas (CountVectorizer)
- Entrena un clasificador Naive Bayes Multinomial
- Ajusta hiperparametros automaticamente con validacion cruzada de 5 pliegues (GridSearchCV)
- Guarda el modelo entrenado en disco con joblib
- Clasifica nuevos mensajes desde un archivo de texto plano, un mensaje por linea

---

## Estructura del proyecto

```
.
├── train_model.py        # Descarga el dataset, entrena el modelo y lo guarda en disco
├── predict.py            # Carga el modelo guardado y clasifica mensajes desde un .txt
├── mensajes_prueba.txt   # Mensajes de ejemplo para pruebas
├── spam_model.joblib     # Modelo guardado (generado despues del entrenamiento)
└── README.md
```

---

## Requisitos

- Python 3.8 o superior
- pip

Instalar dependencias:

```bash
pip install requests pandas scikit-learn nltk joblib
```

---

## Uso

**Paso 1 — Entrenar y guardar el modelo**

Ejecutar una sola vez. Descarga el dataset, entrena el clasificador, imprime un reporte de evaluacion y guarda el modelo.

```bash
python train_model.py
```

**Paso 2 — Clasificar mensajes**

Pasar un archivo `.txt` con un SMS por linea:

```bash
python predict.py mensajes_prueba.txt
```

O ejecutar sin argumentos para clasificar los mensajes de ejemplo integrados:

```bash
python predict.py
```

**Ejemplo de salida:**

```
------------------------------------------------------------
Mensaje  : FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!
Resultado: SPAM
Probabilidad spam: 99.8%  |  Ham: 0.2%
------------------------------------------------------------
Mensaje  : Reminder: Your appointment is scheduled for tomorrow at 10am.
Resultado: HAM (no spam)
Probabilidad spam: 0.4%  |  Ham: 99.6%
------------------------------------------------------------
```

---

## Como funciona

### Pipeline de preprocesamiento

Cada mensaje pasa por los siguientes pasos antes de ser enviado al modelo:

1. Conversion a minusculas
2. Eliminacion de puntuacion y numeros (excepto `$` y `!`, que tienen valor de señal en spam)
3. Tokenizacion
4. Eliminacion de stopwords
5. Stemming con Porter (reduce las palabras a su forma raiz)

### Modelo

Naive Bayes Multinomial es una linea base bien establecida para clasificacion de texto. Modela la probabilidad de que un mensaje pertenezca a cada clase segun la frecuencia de sus terminos. A pesar de su simplicidad, funciona excepcionalmente bien en tareas de deteccion de spam.

El ajuste de hiperparametros se realiza sobre el parametro de suavizado `alpha` usando validacion cruzada de 5 pliegues optimizando el F1-score, que balancea precision y recall.

### Extraccion de features

CountVectorizer convierte los mensajes preprocesados en matrices dispersas de frecuencia de terminos. Se incluyen bigramas (`ngram_range=(1, 2)`) para capturar frases cortas que tienen significado como unidad, como "free entry" o "click here".

---

## Rendimiento

Evaluado via validacion cruzada sobre el dataset completo (5,169 mensajes tras eliminar duplicados):

| Metrica   | Ham   | Spam  |
|-----------|-------|-------|
| Precision | ~0.99 | ~0.97 |
| Recall    | ~0.99 | ~0.96 |
| F1-score  | ~0.99 | ~0.96 |

Los valores reales pueden variar ligeramente entre ejecuciones por el orden del dataset.

---

## Dataset

**SMS Spam Collection** — UCI Machine Learning Repository  
Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering.  
[https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

---

## Conexion con AI Red Teaming

Entender como funcionan los clasificadores de spam es un prerequisito para entender como fallan. El camino AI Red Teamer en Hack The Box cubre tecnicas como:

- Construir entradas adversariales que evadan clasificadores
- Sondear los limites de decision del modelo
- Explotar suposiciones del preprocesamiento (por ejemplo, que pasa si se eliminan los caracteres de los que el modelo depende)
- Entender la brecha entre la distribucion de entrenamiento y las entradas del mundo real

Este proyecto es la linea base del defensor. El siguiente paso es aprender a saltarsela.

---

## Licencia

MIT
