"""
generate_dataset.py
Generates synthetic historical weather data and saves it as
weather_data.csv — used to train the rain prediction model.

Run this ONCE to create the dataset:
    python generate_dataset.py

Columns:
    temperature   — temperature in °C
    humidity      — relative humidity in %
    wind_speed    — wind speed in km/h
    pressure      — surface pressure in hPa
    rain          — 0 (no rain) or 1 (rain) — the TARGET column
"""

import csv
import random
import math

random.seed(42)

N = 500   # number of historical days to generate

rows = []

for i in range(N):

    # ── Generate realistic weather values ─────────────────────────────────────
    # Temperature: 18°C to 42°C  (Mumbai-like tropical climate)
    temperature = round(random.uniform(18, 42), 1)

    # Humidity: 30% to 100%
    humidity = round(random.uniform(30, 100), 1)

    # Wind speed: 0 to 45 km/h
    wind_speed = round(random.uniform(0, 45), 1)

    # Pressure: 990 to 1020 hPa (low = storm, high = clear)
    pressure = round(random.uniform(990, 1020), 1)

    # ── Decide whether it rained — based on realistic rules ──────────────────
    # Rain is MORE LIKELY when:
    #   - Humidity is high (>70%)       → +0.40 points
    #   - Pressure is low  (<1005 hPa)  → +0.30 points
    #   - Temperature is moderate (<30°C) → +0.20 points
    #   - Wind is strong   (>20 km/h)  → +0.10 points
    # Then add random noise so it's not 100% deterministic

    rain_score = 0.0

    if humidity > 70:
        rain_score += 0.40

    if pressure < 1005:
        rain_score += 0.30

    if temperature < 30:
        rain_score += 0.20

    if wind_speed > 20:
        rain_score += 0.10

    # Add random noise (0.0 to 0.30) to make it realistic
    rain_score += random.uniform(0, 0.30)

    # If score > 0.50, it rained that day
    rain = 1 if rain_score > 0.50 else 0

    rows.append({
        "temperature": temperature,
        "humidity":    humidity,
        "wind_speed":  wind_speed,
        "pressure":    pressure,
        "rain":        rain,
    })

# ── Save to CSV ───────────────────────────────────────────────────────────────
filename = "weather_data.csv"
fields   = ["temperature", "humidity", "wind_speed", "pressure", "rain"]

with open(filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# ── Summary ───────────────────────────────────────────────────────────────────
rainy_days  = sum(r["rain"] for r in rows)
clear_days  = N - rainy_days

print(f"Dataset generated: {filename}")
print(f"Total records : {N}")
print(f"Rainy days    : {rainy_days}  ({rainy_days/N*100:.1f}%)")
print(f"Clear days    : {clear_days}  ({clear_days/N*100:.1f}%)")
print()
print("Sample records:")
print("temp  | humidity | wind | pressure | rain")
print("-" * 45)
for row in rows[:5]:
    print(f"{row['temperature']:<6} | {row['humidity']:<8} | "
          f"{row['wind_speed']:<5} | {row['pressure']:<9} | {row['rain']}")
