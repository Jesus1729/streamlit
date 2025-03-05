import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

nltk.download('all')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[_():;,.!?\\-]', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r' +', ' ', text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def train_model(df):
    df['text_clean'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text_clean'])
    y = df['label'].apply(lambda x: 1 if x == 'positive' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Guardar modelo y vectorizador
    os.makedirs('model', exist_ok=True)
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    joblib.dump(model, 'model/model.pkl')
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def predict_sentiment(text):
    vectorizer = joblib.load('model/vectorizer.pkl')
    model = joblib.load('model/model.pkl')
    text_clean = clean_text(text)
    X = vectorizer.transform([text_clean])
    return 'positive' if model.predict(X)[0] == 1 else 'negative'

# Personalización de la app
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            height: 40px;
            width: 160px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>label {
            color: #1e90ff;
            font-weight: bold;
        }
        .stTitle {
            color: #ff6347;
            font-size: 32px;
            text-align: center;
        }
        .stSubheader {
            color: #4682b4;
        }
        .stError {
            color: #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 **Análisis de Sentimientos con Regresión Logística**")

# Subir archivo y mostrar vista previa
uploaded_file = st.file_uploader("Sube un archivo CSV con columnas 'text' y 'label'", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📝 **Vista previa de los datos**:", df.head())
    
    # Entrenar el modelo con color de fondo para el botón
    if st.button("💪 Entrenar modelo"):
        accuracy = train_model(df)
        st.success(f"✅ **Modelo entrenado con precisión: {accuracy:.2%}**")

# Sección para probar el modelo
st.subheader("🔍 **Prueba el modelo**")
user_input = st.text_area("Escribe un texto para analizar el sentimiento:", height=150)
if st.button("⚡ Predecir Sentimiento"):
    if os.path.exists("model/model.pkl"):
        sentiment = predict_sentiment(user_input)
        st.write(f"💬 **Sentimiento Predicho**: **{sentiment}**")
    else:
        st.error("❌ **Primero entrena el modelo subiendo un archivo CSV.**")
