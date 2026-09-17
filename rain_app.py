# app.py
# ============================================================
#  Rain Prediction Dashboard — Brainiac Institute
#  Three tabs: Dataset & Model | Live Prediction | How It Works
# ============================================================

import streamlit as st
import json
import urllib.request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Rain Predictor", page_icon="🌧️", layout="centered")
st.title("🌧️ Rain Prediction System")
st.caption("Brainiac Institute  |  sklearn + Open-Meteo API + Streamlit")
st.divider()


# ── Training data ─────────────────────────────────────────────
X = [
    # temp  humid  wind  pressure     label
    [ 35,    40,    5,   1015 ],   # 0 no rain
    [ 28,    85,   20,    998 ],   # 1 rain
    [ 32,    60,   10,   1010 ],   # 0
    [ 24,    90,   25,    995 ],   # 1
    [ 38,    30,    8,   1018 ],   # 0
    [ 26,    88,   30,    993 ],   # 1
    [ 31,    55,   12,   1008 ],   # 0
    [ 22,    92,   18,    997 ],   # 1
    [ 36,    45,    6,   1016 ],   # 0
    [ 27,    80,   22,   1000 ],   # 1
    [ 33,    50,    9,   1012 ],   # 0
    [ 25,    87,   28,    994 ],   # 1
    [ 34,    42,    7,   1014 ],   # 0
    [ 23,    91,   24,    996 ],   # 1
    [ 30,    58,   11,   1009 ],   # 0
    [ 29,    82,   19,    999 ],   # 1
    [ 37,    35,    5,   1017 ],   # 0
    [ 21,    93,   27,    992 ],   # 1
    [ 32,    48,    8,   1013 ],   # 0
    [ 26,    84,   21,   1001 ],   # 1
]
y = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]


# ── Train model (cached — runs once) ──────────────────────────
@st.cache_resource
def train_model():
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    return clf, acc

model, accuracy = train_model()


# ── Fetch live weather ────────────────────────────────────────
def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,surface_pressure"
        f"&timezone=Asia%2FKolkata"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            d = json.loads(r.read())["current"]
        return {"temperature": d["temperature_2m"],
                "humidity":    d["relative_humidity_2m"],
                "wind_speed":  d["wind_speed_10m"],
                "pressure":    d["surface_pressure"]}, None
    except Exception as e:
        return None, str(e)


# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dataset & Model","🌤️ Live Prediction","🧠 How It Works"])


# TAB 1 — Dataset & Model
with tab1:
    st.subheader("Training Dataset — 20 Past Weather Records")
    cols = ["Temperature (°C)","Humidity (%)","Wind (km/h)","Pressure (hPa)"]
    df = pd.DataFrame(X, columns=cols)
    df["Rain"] = ["🌧️ Yes" if v else "☀️ No" for v in y]
    st.dataframe(df, use_container_width=True)
    st.divider()
    st.metric("Model Accuracy (RandomForest)", f"{accuracy:.0%}")
    st.caption("Trained on 16 rows, tested on 4 rows. More data = better accuracy.")


# TAB 2 — Live Prediction
with tab2:
    st.subheader("Live Weather Prediction")

    cities = {
        "Mumbai"   :(19.08, 72.88), "Delhi"    :(28.61, 77.21),
        "Bangalore":(12.97, 77.59), "Chennai"  :(13.08, 80.27),
        "Kolkata"  :(22.57, 88.36), "Hyderabad":(17.39, 78.49),
        "Pune"     :(18.52, 73.86),
    }

    city = st.selectbox("Select your city", list(cities.keys()))
    lat, lon = cities[city]

    if st.button("🌐 Fetch Weather and Predict", type="primary", use_container_width=True):
        with st.spinner(f"Fetching live weather for {city}..."):
            weather, err = fetch_weather(lat, lon)

        if err:
            st.error(f"Could not fetch weather: {err}")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🌡️ Temp",    f"{weather['temperature']}°C")
            c2.metric("💧 Humidity", f"{weather['humidity']}%")
            c3.metric("🌬️ Wind",     f"{weather['wind_speed']} km/h")
            c4.metric("🔵 Pressure", f"{weather['pressure']} hPa")
            st.divider()

            live = [[weather["temperature"], weather["humidity"],
                     weather["wind_speed"],  weather["pressure"]]]

            pred  = model.predict(live)[0]
            proba = model.predict_proba(live)[0]
            rain  = round(proba[1] * 100, 1)

            if pred == 1:
                st.error(f"## 🌧️ RAIN PREDICTED\nRain probability: **{rain}%**\n\n☂️ Carry an umbrella!")
            else:
                st.success(f"## ☀️ NO RAIN\nRain probability: **{rain}%**\n\n😊 Clear day ahead!")

            st.markdown("**Rain probability:**")
            st.progress(rain / 100)


# TAB 3 — How It Works
with tab3:
    st.subheader("How This System Works")
    st.markdown("""
| Step | What happens | Tool |
|------|-------------|------|
| 1 | Weather data is defined as Python lists | Python |
| 2 | Model is trained on past data | sklearn RandomForest |
| 3 | Live weather is fetched | Open-Meteo API (free) |
| 4 | Live weather is fed into the model | sklearn predict() |
| 5 | Result is sent to Telegram | Telegram Bot API |
| 6 | Steps 3–5 run every morning automatically | GitHub Actions |
    """)

    st.divider()
    st.subheader("GitHub Secrets and Variables")
    st.code("""
# Secrets (Settings → Secrets → Actions)
TELEGRAM_BOT_TOKEN   → from @BotFather on Telegram
TELEGRAM_CHAT_ID     → from api.telegram.org/bot{TOKEN}/getUpdates

# Variables (Settings → Variables → Actions)
CITY       → e.g. Kolkata
LATITUDE   → e.g. 22.57
LONGITUDE  → e.g. 88.36
    """, language="text")

st.divider()
st.caption("Brainiac Institute  |  Rain Predictor  |  sklearn + Streamlit")
