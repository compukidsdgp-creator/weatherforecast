# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Rain Prediction Dashboard — Brainiac Institute
# Streamlit app that shows:
#   1. The training dataset and its patterns
#   2. Model training accuracy
#   3. Live prediction with any custom input
#   4. How each feature affects the prediction
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import csv
import math
import json
import urllib.request
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain Prediction System",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Rain Prediction System")
st.caption("Brainiac Institute  |  Logistic Regression  |  Live Weather + ML")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS — same as predict_rain.py
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath):
    X, y = [], []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row["temperature"]), float(row["humidity"]),
                      float(row["wind_speed"]),  float(row["pressure"])])
            y.append(int(row["rain"]))
    return X, y

def normalize(X):
    n_features = len(X[0])
    mins  = [min(r[j] for r in X) for j in range(n_features)]
    maxs  = [max(r[j] for r in X) for j in range(n_features)]
    X_norm = [[(r[j]-mins[j])/(maxs[j]-mins[j]) if maxs[j]!=mins[j] else 0
               for j in range(n_features)] for r in X]
    return X_norm, mins, maxs

def norm_row(row, mins, maxs):
    return [(row[j]-mins[j])/(maxs[j]-mins[j]) if maxs[j]!=mins[j] else 0
            for j in range(len(row))]

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-500, min(500, z))))

def dot(w, x):
    return sum(a*b for a,b in zip(w, x))

def train_lr(X, y, lr=0.1, epochs=300):
    n, nf  = len(X), len(X[0])
    w, b   = [0.0]*nf, 0.0
    losses = []
    for _ in range(epochs):
        dw, db = [0.0]*nf, 0.0
        loss   = 0
        for i in range(n):
            yh    = sigmoid(dot(w, X[i]) + b)
            err   = yh - y[i]
            loss += -(y[i]*math.log(max(yh,1e-9))+(1-y[i])*math.log(max(1-yh,1e-9)))
            for j in range(nf): dw[j] += err * X[i][j]
            db += err
        for j in range(nf): w[j] -= lr * dw[j] / n
        b -= lr * db / n
        losses.append(loss / n)
    return w, b, losses

def predict_proba(w, b, row):
    return sigmoid(dot(w, row) + b)

def calc_accuracy(X, y, w, b):
    return sum(1 for i in range(len(X))
               if (1 if predict_proba(w,b,X[i])>=0.5 else 0)==y[i]) / len(X)

def fetch_weather(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,"
           f"wind_speed_10m,surface_pressure,precipitation"
           f"&timezone=Asia/Kolkata")
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())["current"]
        return {
            "temperature": round(data["temperature_2m"], 1),
            "humidity":    round(data["relative_humidity_2m"], 1),
            "wind_speed":  round(data["wind_speed_10m"], 1),
            "pressure":    round(data["surface_pressure"], 1),
            "precipitation": round(data.get("precipitation", 0), 1),
        }, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & TRAIN (cached so it runs only once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    X, y = load_dataset("weather_data.csv")
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    X_tr_n, mins, maxs = normalize(X_tr)
    X_te_n = [norm_row(r, mins, maxs) for r in X_te]
    w, b, losses = train_lr(X_tr_n, y_tr)
    tr_acc = calc_accuracy(X_tr_n, y_tr, w, b)
    te_acc = calc_accuracy(X_te_n, y_te, w, b)
    return X, y, w, b, mins, maxs, losses, tr_acc, te_acc

try:
    X, y, weights, bias, feat_mins, feat_maxs, losses, train_acc, test_acc = load_and_train()
    dataset_ok = True
except FileNotFoundError:
    st.error("❌ `weather_data.csv` not found. Run `python generate_dataset.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
city      = st.sidebar.text_input("City Name (display only)", "Mumbai")
latitude  = st.sidebar.number_input("Latitude",  value=19.08, step=0.01, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=72.88, step=0.01, format="%.4f")
threshold = st.sidebar.slider("Rain Alert Threshold", 0.1, 0.9, 0.50, 0.05)
st.sidebar.divider()
st.sidebar.info(
    "**City Coordinates (examples)**\n\n"
    "Mumbai: 19.08, 72.88\n"
    "Delhi: 28.61, 77.21\n"
    "Bangalore: 12.97, 77.59\n"
    "Chennai: 13.08, 80.27\n"
    "Kolkata: 22.57, 88.36"
)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Dataset & Model", "🌤️ Live Prediction", "🧠 How It Works"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATASET & MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Training Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(X))
    col2.metric("Rainy Days",    f"{sum(y)}  ({sum(y)/len(y)*100:.0f}%)")
    col3.metric("Clear Days",    f"{len(y)-sum(y)}  ({(len(y)-sum(y))/len(y)*100:.0f}%)")

    # Show a sample of the raw data as a table
    st.markdown("**Sample records from `weather_data.csv`:**")
    headers = ["#", "Temperature (°C)", "Humidity (%)", "Wind (km/h)", "Pressure (hPa)", "Rain"]
    sample  = [[i+1, X[i][0], X[i][1], X[i][2], X[i][3],
                "🌧️ Yes" if y[i] else "☀️ No"] for i in range(10)]
    st.table([dict(zip(headers, r)) for r in sample])

    st.divider()
    st.subheader("Model Training Results")

    c1, c2 = st.columns(2)
    c1.metric("Training Accuracy", f"{train_acc:.1%}")
    c2.metric("Testing Accuracy",  f"{test_acc:.1%}")

    # Loss curve
    st.markdown("**Training Loss Curve** — how the model improves over 300 epochs:")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(losses, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Logistic Regression — Loss Over Training Epochs")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Loss should decrease and flatten — this shows the model is learning.")

    st.divider()
    st.subheader("Feature Weights — What the Model Learned")
    feature_names = ["Temperature", "Humidity", "Wind Speed", "Pressure"]
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    colors = ["tomato" if w > 0 else "steelblue" for w in weights]
    ax2.barh(feature_names, weights, color=colors, edgecolor="white", height=0.5)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Feature Weights (Positive = increases rain probability)")
    ax2.set_xlabel("Weight Value")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption(
        "🔴 Red = positive weight (feature increases rain chance). "
        "🔵 Blue = negative weight (feature decreases rain chance)."
    )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🌤️ Live Weather Prediction for {city}")

    col_live, col_manual = st.columns([1, 1])

    with col_live:
        st.markdown("**Option A — Fetch live weather automatically**")
        if st.button("🌐 Fetch Live Weather"):
            with st.spinner(f"Fetching weather for {city}..."):
                live, err = fetch_weather(latitude, longitude)
            if live:
                st.session_state["weather"] = live
                st.success("Live weather fetched!")
            else:
                st.error(f"Could not fetch: {err}")
                st.info("Use Option B to enter values manually.")

        if "weather" in st.session_state:
            w = st.session_state["weather"]
            st.metric("Temperature",  f"{w['temperature']} °C")
            st.metric("Humidity",     f"{w['humidity']} %")
            st.metric("Wind Speed",   f"{w['wind_speed']} km/h")
            st.metric("Pressure",     f"{w['pressure']} hPa")
            st.metric("Precipitation",f"{w['precipitation']} mm")

    with col_manual:
        st.markdown("**Option B — Enter values manually**")
        temp_in  = st.number_input("Temperature (°C)", 10.0, 50.0, 30.0, 0.1)
        hum_in   = st.number_input("Humidity (%)",      10.0, 100.0, 70.0, 1.0)
        wind_in  = st.number_input("Wind Speed (km/h)",  0.0, 60.0, 15.0, 0.5)
        pres_in  = st.number_input("Pressure (hPa)",    970.0, 1030.0, 1005.0, 0.5)

    st.divider()

    # Predict button
    if st.button("🔮 Predict Rain", type="primary", use_container_width=True):

        # Use live weather if available, else manual
        if "weather" in st.session_state:
            w = st.session_state["weather"]
            inp = [w["temperature"], w["humidity"], w["wind_speed"], w["pressure"]]
            source = "Live weather data"
        else:
            inp    = [temp_in, hum_in, wind_in, pres_in]
            source = "Manual input"

        # Normalise and predict
        inp_norm  = norm_row(inp, feat_mins, feat_maxs)
        rain_prob = predict_proba(weights, bias, inp_norm)
        prediction= 1 if rain_prob >= threshold else 0
        pct       = rain_prob * 100

        st.subheader("Prediction Result")
        st.caption(f"Source: {source}")

        # Progress bar for probability
        st.markdown(f"**Rain Probability: {pct:.1f}%**")
        st.progress(rain_prob)

        # Verdict
        if prediction == 1:
            st.error(
                f"🌧️ **RAIN PREDICTED** in {city} today!\n\n"
                f"Probability: **{pct:.1f}%** (above {threshold:.0%} threshold)\n\n"
                f"☂️ Carry an umbrella and avoid long outdoor trips."
            )
        else:
            st.success(
                f"☀️ **NO RAIN** expected in {city} today.\n\n"
                f"Probability: **{pct:.1f}%** (below {threshold:.0%} threshold)\n\n"
                f"😊 Enjoy the day — looks clear!"
            )

        # Input summary
        st.markdown("**Input values used:**")
        st.table({
            "Feature"    : ["Temperature", "Humidity", "Wind Speed", "Pressure"],
            "Value"      : [f"{inp[0]}°C", f"{inp[1]}%", f"{inp[2]} km/h", f"{inp[3]} hPa"],
            "Normalised" : [f"{inp_norm[j]:.3f}" for j in range(4)],
        })

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧠 How This System Works")

    st.markdown("""
    This project combines **four things you have already learned** into one automated system:

    | Step | What happens | Technology |
    |------|-------------|-----------|
    | 1 | Generate synthetic historical weather data | Python (csv, random) |
    | 2 | Train a Logistic Regression model on it | Pure Python (no sklearn) |
    | 3 | Fetch today's live weather | Open-Meteo API (free) |
    | 4 | Predict rain using the trained model | Logistic Regression |
    | 5 | Send the result to Telegram | Telegram Bot API |
    | 6 | Run steps 3–5 automatically every morning | GitHub Actions |
    """)

    st.divider()
    st.subheader("The Logistic Regression Formula")
    st.latex(r"P(\text{rain}) = \sigma(w_1 \cdot \text{temp} + w_2 \cdot \text{humidity} + w_3 \cdot \text{wind} + w_4 \cdot \text{pressure} + b)")
    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
    st.markdown("""
    - **σ (sigmoid)** converts any number into a probability between 0 and 1
    - **w₁, w₂, w₃, w₄** are the weights the model learns during training
    - **b** is the bias term
    - If the result ≥ 0.50 → predict Rain; otherwise → predict No Rain
    """)

    st.divider()
    st.subheader("Project File Structure")
    st.code("""
rain_prediction/
│
├── generate_dataset.py     # Creates weather_data.csv (run once)
├── weather_data.csv        # 500-day synthetic training dataset
├── predict_rain.py         # Main script — train, fetch, predict, send
├── app.py                  # This Streamlit dashboard
├── requirements.txt        # streamlit, matplotlib (only 2 packages)
└── .github/
    └── workflows/
        └── rain_prediction.yml   # GitHub Actions — runs daily at 7am IST
    """, language="text")

    st.divider()
    st.subheader("GitHub Secrets Required")
    st.table({
        "Secret Name"         : ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"],
        "Where to get it"     : ["@BotFather on Telegram", "getUpdates API call"],
        "Type"                : ["Secret", "Secret"],
    })
    st.subheader("GitHub Variables (optional)")
    st.table({
        "Variable Name"  : ["CITY", "LATITUDE", "LONGITUDE", "RAIN_THRESHOLD"],
        "Default"        : ["Mumbai", "19.08", "72.88", "0.50"],
        "Type"           : ["Variable", "Variable", "Variable", "Variable"],
    })

st.divider()
st.caption("Brainiac Institute  |  Rain Prediction System  |  Logistic Regression + GitHub Actions + Telegram")
