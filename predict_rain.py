# predict_rain.py
# ============================================================
#  Simple Rain Predictor
#  Brainiac Institute
#
#  What this script does — in order:
#  1. Create a small weather dataset
#  2. Train a model using sklearn
#  3. Fetch today's live weather
#  4. Predict: Rain or No Rain
#  5. Send result to Telegram
# ============================================================

import os
import json
import urllib.request
import urllib.error
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# STEP 1 — TRAINING DATA
# Past weather observations with known outcomes.
# Each row: [Temperature, Humidity, Wind Speed, Pressure]
# Label   :  1 = Rained    0 = Did not rain
# ------------------------------------------------------------

X = [
    # temp  humid  wind  pressure
    [ 35,    40,    5,   1015 ],   # hot, dry       → no rain
    [ 28,    85,   20,    998 ],   # humid, windy   → rain
    [ 32,    60,   10,   1010 ],   # moderate       → no rain
    [ 24,    90,   25,    995 ],   # very humid     → rain
    [ 38,    30,    8,   1018 ],   # very hot, dry  → no rain
    [ 26,    88,   30,    993 ],   # stormy         → rain
    [ 31,    55,   12,   1008 ],   # warm           → no rain
    [ 22,    92,   18,    997 ],   # cool, humid    → rain
    [ 36,    45,    6,   1016 ],   # hot            → no rain
    [ 27,    80,   22,   1000 ],   # humid          → rain
    [ 33,    50,    9,   1012 ],   # warm, dry      → no rain
    [ 25,    87,   28,    994 ],   # humid, windy   → rain
    [ 34,    42,    7,   1014 ],   # hot, dry       → no rain
    [ 23,    91,   24,    996 ],   # cool, humid    → rain
    [ 30,    58,   11,   1009 ],   # moderate       → no rain
    [ 29,    82,   19,    999 ],   # humid          → rain
    [ 37,    35,    5,   1017 ],   # very hot, dry  → no rain
    [ 21,    93,   27,    992 ],   # cool, rainy    → rain
    [ 32,    48,    8,   1013 ],   # warm           → no rain
    [ 26,    84,   21,   1001 ],   # humid          → rain
]

y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


# ------------------------------------------------------------
# STEP 2 — TRAIN THE MODEL
# Split data → train model → check accuracy
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42
)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"[1] Model trained   — Accuracy: {accuracy:.0%}")


# ------------------------------------------------------------
# STEP 3 — FETCH TODAY'S LIVE WEATHER
# Open-Meteo API: completely free, no API key needed.
# Change CITY / LATITUDE / LONGITUDE via GitHub Variables.
# ------------------------------------------------------------

CITY      = os.environ.get("CITY",      "").strip() or "Mumbai"
LATITUDE  = os.environ.get("LATITUDE",  "").strip() or "19.08"
LONGITUDE = os.environ.get("LONGITUDE", "").strip() or "72.88"

url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&current=temperature_2m,relative_humidity_2m,"
    f"wind_speed_10m,surface_pressure"
    f"&timezone=Asia%2FKolkata"
)

try:
    with urllib.request.urlopen(url, timeout=10) as response:
        data    = json.loads(response.read())
        current = data["current"]

    temperature = current["temperature_2m"]
    humidity    = current["relative_humidity_2m"]
    wind_speed  = current["wind_speed_10m"]
    pressure    = current["surface_pressure"]
    print(f"[2] Live weather    — Temp:{temperature}°C  "
          f"Humidity:{humidity}%  "
          f"Wind:{wind_speed}km/h  "
          f"Pressure:{pressure}hPa")

except Exception as e:
    print(f"[2] Weather fetch failed ({e}) — using test values")
    temperature, humidity, wind_speed, pressure = 27.0, 85.0, 22.0, 998.0


# ------------------------------------------------------------
# STEP 4 — PREDICT RAIN OR NO RAIN
# ------------------------------------------------------------

today_weather = [[temperature, humidity, wind_speed, pressure]]

prediction  = model.predict(today_weather)[0]
probability = model.predict_proba(today_weather)[0]
rain_chance = round(probability[1] * 100, 1)

if prediction == 1:
    result = "🌧️ 😎 RAIN PREDICTED"
    advice = "Carry an umbrella. Avoid outdoor plans if possible."
else:
    result = "☀️ NO RAIN"
    advice = "Looks clear! Good day for outdoor activities."

print(f"[3] Prediction      — {result}  (Rain chance: {rain_chance}%)")


# ------------------------------------------------------------
# STEP 5 — BUILD THE TELEGRAM MESSAGE
# ------------------------------------------------------------

today = datetime.now().strftime("%d %B %Y, %I:%M %p")

message = f"""
\U0001f326\ufe0f *DAILY WEATHER REPORT*
\U0001f4cd {CITY}
\U0001f4c5 {today}
{'─' * 28}

\U0001f321\ufe0f Temperature : {temperature}\u00b0C
\U0001f4a7 Humidity    : {humidity}%
\U0001f32c\ufe0f Wind Speed  : {wind_speed} km/h
\U0001f535 Pressure    : {pressure} hPa
{'─' * 28}

\U0001f4ca Rain Chance : *{rain_chance}%*
\U0001f52e Prediction  : *{result}*

\U0001f4ac {advice}
{'─' * 28}
\U0001f916 _Sent by GitHub Actions_
""".strip()

print("[4] Message built")


# ------------------------------------------------------------
# STEP 6 — SEND TO TELEGRAM
# Reads token and chat ID from GitHub Secrets.
# ------------------------------------------------------------

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID",   "").strip()

if not TOKEN or not CHAT_ID:
    print("[5] Telegram skipped — no credentials found")
    print("    Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as GitHub Secrets")
    print("\n--- MESSAGE PREVIEW ---")
    print(message)

else:
    api_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id"    : CHAT_ID,
        "text"       : message,
        "parse_mode" : "Markdown",
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read())
        if res.get("ok"):
            print("[5] Telegram sent ✅")
        else:
            print(f"[5] Telegram error — {res.get('description')}")

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:   desc = json.loads(body).get("description", e.reason)
        except Exception: desc = e.reason
        print(f"[5] Failed — HTTP {e.code}: {desc}")
        if e.code == 404: print("    → Token wrong. Check TELEGRAM_BOT_TOKEN.")
        elif e.code == 400: print("    → Chat ID wrong. Check TELEGRAM_CHAT_ID.")
        elif e.code == 403: print("    → Use YOUR chat ID, not the bot's ID.")
