# predict_rain.py
# ─────────────────────────────────────────────────────────────────────────────
# Rain Prediction System — Brainiac Institute
#
# What this script does:
#   1. Loads the synthetic historical weather dataset (weather_data.csv)
#   2. Trains a Logistic Regression model to predict rain
#   3. Fetches TODAY's live weather from Open-Meteo API (free, no API key)
#   4. Uses the trained model to predict whether it will rain today
#   5. Sends a Telegram message with the prediction and weather details
#
# Required GitHub Secrets:
#   TELEGRAM_BOT_TOKEN  — your bot token from @BotFather
#   TELEGRAM_CHAT_ID    — your chat or group ID
#
# Optional GitHub Variables:
#   CITY         — city name for display (default: Mumbai)
#   LATITUDE     — city latitude  (default: 19.08 = Mumbai)
#   LONGITUDE    — city longitude (default: 72.88 = Mumbai)
#   RAIN_THRESHOLD — probability threshold to trigger alert (default: 0.5)
#
# To run locally for testing (prints prediction, skips Telegram if no token):
#   python predict_rain.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import json
import math
import urllib.request
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — reads from environment variables (GitHub Secrets/Variables)
# ══════════════════════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID        = os.environ.get("TELEGRAM_CHAT_ID",   "")
CITY           = os.environ.get("CITY",               "")
LATITUDE       = float(os.environ.get("LATITUDE",     ""))
LONGITUDE      = float(os.environ.get("LONGITUDE",    ""))
RAIN_THRESHOLD = float(os.environ.get("RAIN_THRESHOLD",""))
DATASET_FILE   = "weather_data.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD THE DATASET
#  Reads weather_data.csv into a list of dictionaries
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(filepath):
    """
    Loads the CSV dataset into a list of dicts.
    Converts all numeric columns to float.
    Returns X (features) and y (labels) as separate lists.
    """
    X = []   # features: [[temp, humidity, wind, pressure], ...]
    y = []   # labels:   [0, 1, 1, 0, ...]

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([
                float(row["temperature"]),
                float(row["humidity"]),
                float(row["wind_speed"]),
                float(row["pressure"]),
            ])
            y.append(int(row["rain"]))

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — LOGISTIC REGRESSION (from scratch — no sklearn)
#  We implement Logistic Regression manually so students can see
#  exactly how it works: sigmoid function + gradient descent.
# ══════════════════════════════════════════════════════════════════════════════

def normalize(X):
    """
    Normalises features to range [0, 1] using Min-Max scaling.
    Returns normalised X, and the min/max values for each feature
    (needed to normalise new input data the same way).
    """
    n_features = len(X[0])
    mins  = [min(row[j] for row in X) for j in range(n_features)]
    maxs  = [max(row[j] for row in X) for j in range(n_features)]

    X_norm = []
    for row in X:
        norm_row = []
        for j in range(n_features):
            rng = maxs[j] - mins[j]
            val = (row[j] - mins[j]) / rng if rng != 0 else 0
            norm_row.append(val)
        X_norm.append(norm_row)

    return X_norm, mins, maxs


def sigmoid(z):
    """Sigmoid function: converts any number to a probability (0 to 1)."""
    return 1 / (1 + math.exp(-max(-500, min(500, z))))


def dot(weights, row):
    """Dot product: weights · features."""
    return sum(w * x for w, x in zip(weights, row))


def train_logistic_regression(X, y, lr=0.1, epochs=200):
    """
    Trains Logistic Regression using Gradient Descent.

    Parameters:
        X      — list of feature rows (normalised)
        y      — list of labels (0 or 1)
        lr     — learning rate (how big each update step is)
        epochs — number of training iterations

    Returns:
        weights — learned weights for each feature
        bias    — learned bias term
    """
    n = len(X)
    n_features = len(X[0])

    # Start weights and bias at 0
    weights = [0.0] * n_features
    bias    = 0.0

    for epoch in range(epochs):
        # Accumulate gradients
        dw = [0.0] * n_features
        db = 0.0

        for i in range(n):
            # Forward pass: predict probability
            z     = dot(weights, X[i]) + bias
            y_hat = sigmoid(z)

            # Error
            error = y_hat - y[i]

            # Accumulate gradient
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error

        # Update weights and bias
        for j in range(n_features):
            weights[j] -= lr * dw[j] / n
        bias -= lr * db / n

    return weights, bias


def predict_proba(weights, bias, row):
    """Returns the probability of rain (0.0 to 1.0) for one row."""
    z = dot(weights, row) + bias
    return sigmoid(z)


def accuracy(X, y, weights, bias):
    """Calculates model accuracy on a dataset."""
    correct = 0
    for i in range(len(X)):
        prob = predict_proba(weights, bias, X[i])
        pred = 1 if prob >= 0.5 else 0
        if pred == y[i]:
            correct += 1
    return correct / len(X)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FETCH LIVE WEATHER FROM OPEN-METEO API
#  Free API, no key needed, works from GitHub Actions and any machine.
#  Docs: https://open-meteo.com/en/docs
# ══════════════════════════════════════════════════════════════════════════════

def fetch_weather(latitude, longitude, city):
    """
    Fetches current weather from Open-Meteo API.
    Returns a dict with temperature, humidity, wind_speed, pressure.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&current=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,surface_pressure,precipitation"
        f"&timezone=Asia/Kolkata"
    )

    print(f"Fetching live weather for {city}...")
    print(f"URL: {url}")

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())

        current = data["current"]

        weather = {
            "temperature": round(current["temperature_2m"], 1),
            "humidity":    round(current["relative_humidity_2m"], 1),
            "wind_speed":  round(current["wind_speed_10m"], 1),
            "pressure":    round(current["surface_pressure"], 1),
            "precipitation": round(current.get("precipitation", 0), 1),
            "fetched_at":  datetime.now().strftime("%d %b %Y, %I:%M %p"),
        }

        print(f"Live weather: {weather}")
        return weather

    except Exception as e:
        print(f"Warning: Could not fetch live weather — {e}")
        print("Using fallback test values for demonstration.")
        # Fallback values for testing (high humidity = should predict rain)
        return {
            "temperature": 27.5,
            "humidity":    85.0,
            "wind_speed":  22.0,
            "pressure":    998.5,
            "precipitation": 0.0,
            "fetched_at":  datetime.now().strftime("%d %b %Y, %I:%M %p"),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — BUILD THE TELEGRAM MESSAGE
# ══════════════════════════════════════════════════════════════════════════════

def build_message(city, weather, rain_prob, prediction, threshold):
    """
    Builds a formatted Telegram message based on prediction result.
    """
    today = datetime.now().strftime("%A, %d %B %Y")
    pct   = round(rain_prob * 100, 1)

    # Probability bar (visual indicator)
    filled = int(pct / 10)
    bar    = "🟦" * filled + "⬜" * (10 - filled)

    if prediction == 1:
        header  = "🌧️ *RAIN PREDICTED TODAY*"
        verdict = f"⚠️ There is a *{pct}%* chance of rain in {city} today."
        advice  = (
            "☂️ *Recommendations:*\n"
            "  • Carry an umbrella\n"
            "  • Avoid outdoor activities if possible\n"
            "  • Check roads for waterlogging\n"
            "  • Keep an extra set of clothes handy"
        )
    else:
        header  = "☀️ *CLEAR WEATHER TODAY*"
        verdict = f"✅ Rain probability is only *{pct}%* in {city} — looks clear!"
        advice  = (
            "😊 *Recommendations:*\n"
            "  • Great day for outdoor activities\n"
            "  • Light clothing recommended\n"
            "  • Stay hydrated — it may be warm"
        )

    message = f"""
{header}
📍 {city}  |  📅 {today}
⏰ Data fetched at: {weather['fetched_at']}
{'─' * 32}

🌡️ Temperature  : {weather['temperature']}°C
💧 Humidity     : {weather['humidity']}%
🌬️ Wind Speed   : {weather['wind_speed']} km/h
🔵 Pressure     : {weather['pressure']} hPa
🌂 Precipitation: {weather['precipitation']} mm
{'─' * 32}

📊 *Rain Probability*
{bar}  {pct}%
_(Alert threshold: {int(threshold*100)}%)_

{verdict}

{advice}
{'─' * 32}
🤖 _Predicted by Logistic Regression_
_Trained on 500 days of historical data_
_Automated via GitHub Actions_
""".strip()

    return message


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SEND TELEGRAM MESSAGE
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(token, chat_id, message):
    """
    Sends a message using the Telegram Bot API.
    Includes detailed error output so problems are easy to diagnose.

    404 error usually means:  wrong bot token (check GitHub Secrets)
    400 error usually means:  wrong chat_id, or Markdown formatting issue
    401 error usually means:  bot token is invalid or revoked
    """

    # ── Debug: print what we are about to send ────────────────────────────────
    print(f"    Bot token (first 10 chars) : {token[:10]}...")
    print(f"    Chat ID                    : {chat_id}")

    # ── First attempt: send with Markdown formatting ──────────────────────────
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def _post(text, parse_mode=None):
        payload_dict = {
            "chat_id" : str(chat_id).strip(),
            "text"    : text,
        }
        if parse_mode:
            payload_dict["parse_mode"] = parse_mode

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read()), None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            return None, f"HTTP {e.code}: {e.reason} — {body}"
        except Exception as e:
            return None, str(e)

    # Attempt 1: with Markdown
    result, error = _post(message, parse_mode="Markdown")

    if result and result.get("ok"):
        print("✅ Telegram message sent (Markdown mode).")
        return

    # Attempt 2: Markdown failed — strip formatting and send as plain text
    print(f"    Markdown send failed: {error}")
    print("    Retrying as plain text (no formatting)...")

    plain_message = (message
        .replace("*", "")
        .replace("_", "")
        .replace("`", "")
        .replace("[", "")
        .replace("]", ""))

    result, error = _post(plain_message, parse_mode=None)

    if result and result.get("ok"):
        print("✅ Telegram message sent (plain text mode).")
        return

    # Both attempts failed — print full diagnostic and raise
    print(f"\n❌ Telegram send failed on both attempts.")
    print(f"   Last error : {error}")
    print(f"\n── Diagnosis ──────────────────────────────────────────")
    print(f"   If error is 404 → Bot token is WRONG or has extra spaces.")
    print(f"       Check GitHub Secret: TELEGRAM_BOT_TOKEN")
    print(f"       Token format looks like: 7123456789:AAFxxxxx...")
    print(f"   If error is 400 → Chat ID is WRONG.")
    print(f"       Check GitHub Secret: TELEGRAM_CHAT_ID")
    print(f"       Personal ID is a positive integer.")
    print(f"       Group ID is a negative integer (e.g. -987654321).")
    print(f"   If error is 401 → Token is revoked.")
    print(f"       Go to @BotFather → /revoke → generate a new token.")
    print(f"───────────────────────────────────────────────────────")

    raise Exception(f"Telegram send failed: {error}")



# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — runs everything in order
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("  Rain Prediction System — Brainiac Institute")
    print(f"  {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    print("=" * 50)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("\n[1] Loading dataset...")
    X, y = load_dataset(DATASET_FILE)
    print(f"    Loaded {len(X)} records from {DATASET_FILE}")

    # ── 2. Split into train/test (80/20) ──────────────────────────────────────
    print("\n[2] Splitting dataset (80% train / 20% test)...")
    split  = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    print(f"    Training samples : {len(X_train)}")
    print(f"    Testing  samples : {len(X_test)}")

    # ── 3. Normalise features ─────────────────────────────────────────────────
    print("\n[3] Normalising features (Min-Max scaling)...")
    X_train_norm, feat_mins, feat_maxs = normalize(X_train)

    # Normalise test set using SAME min/max as training set
    X_test_norm = []
    for row in X_test:
        norm_row = []
        for j in range(len(row)):
            rng = feat_maxs[j] - feat_mins[j]
            val = (row[j] - feat_mins[j]) / rng if rng != 0 else 0
            norm_row.append(val)
        X_test_norm.append(norm_row)

    # ── 4. Train model ────────────────────────────────────────────────────────
    print("\n[4] Training Logistic Regression model...")
    weights, bias = train_logistic_regression(
        X_train_norm, y_train, lr=0.1, epochs=300
    )

    train_acc = accuracy(X_train_norm, y_train, weights, bias)
    test_acc  = accuracy(X_test_norm,  y_test,  weights, bias)
    print(f"    Training accuracy : {train_acc:.2%}")
    print(f"    Testing  accuracy : {test_acc:.2%}")

    # ── 5. Fetch live weather ─────────────────────────────────────────────────
    print(f"\n[5] Fetching live weather for {CITY}...")
    weather = fetch_weather(LATITUDE, LONGITUDE, CITY)

    # ── 6. Normalise live input using same min/max ────────────────────────────
    live_input = [
        weather["temperature"],
        weather["humidity"],
        weather["wind_speed"],
        weather["pressure"],
    ]
    live_norm = []
    for j in range(len(live_input)):
        rng = feat_maxs[j] - feat_mins[j]
        val = (live_input[j] - feat_mins[j]) / rng if rng != 0 else 0
        live_norm.append(val)

    # ── 7. Predict ────────────────────────────────────────────────────────────
    print("\n[6] Predicting rain probability...")
    rain_prob  = predict_proba(weights, bias, live_norm)
    prediction = 1 if rain_prob >= RAIN_THRESHOLD else 0

    print(f"    Live input   : temp={live_input[0]}°C  "
          f"humidity={live_input[1]}%  "
          f"wind={live_input[2]}km/h  "
          f"pressure={live_input[3]}hPa")
    print(f"    Rain prob    : {rain_prob:.2%}")
    print(f"    Threshold    : {RAIN_THRESHOLD:.0%}")
    print(f"    Prediction   : {'🌧️ RAIN' if prediction else '☀️ NO RAIN'}")

    # ── 8. Build message ──────────────────────────────────────────────────────
    message = build_message(CITY, weather, rain_prob, prediction, RAIN_THRESHOLD)
    print("\n--- MESSAGE PREVIEW ---")
    print(message)
    print("-" * 50)

    # ── 9. Send to Telegram ───────────────────────────────────────────────────
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("\n⚠️  TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        print("    Add them as GitHub Secrets to enable Telegram alerts.")
        print("    Message preview printed above.")
        return

    # Validate token format — Telegram tokens look like: 1234567890:AAFxxx...
    if ":" not in TELEGRAM_TOKEN:
        print("\n❌ TELEGRAM_BOT_TOKEN looks wrong — it must contain a colon (:)")
        print("   Correct format:  7123456789:AAFxxxxxxxxxxxxxxxxxxxxx")
        print("   Check your GitHub Secret for extra spaces or missing characters.")
        raise Exception("Invalid bot token format")

    print("\n[7] Sending Telegram message...")
    send_telegram(TELEGRAM_TOKEN, CHAT_ID, message)
    print("\n✅ Rain prediction complete!")


if __name__ == "__main__":
    main()
