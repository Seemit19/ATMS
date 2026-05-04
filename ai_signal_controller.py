"""
🚦 AI Traffic Signal Controller (ATMS)
--------------------------------------
This script connects the trained congestion prediction model
to an adaptive signal decision system.

Features:
- Loads trained model & encoder
- Reads or simulates live data
- Predicts congestion probability for each direction
- Dynamically allocates green duration
- Logs decision and congestion per cycle

Author: Seemit Kumar & Team
"""

import joblib
import json
import pandas as pd
import numpy as np
import os
import time
import random
from datetime import datetime

# -------------------------------
# Load Trained Model + Encoder
# -------------------------------
print("📦 Loading model and encoder...")
model = joblib.load("rf_multi_congestion_model(3).joblib")
encoder = joblib.load("encoder.joblib")
with open("rf_model_features(3).json") as f:
    feature_info = json.load(f)
feature_columns = feature_info["columns"]

print("✅ Model and encoder loaded successfully.\n")

# -------------------------------
# Simulate Live Traffic Data
# (Replace this with sensor or SUMO input later)
# -------------------------------
def get_live_data():
    data = {
        "count_N": np.random.randint(5, 120),
        "count_S": np.random.randint(5, 120),
        "count_E": np.random.randint(5, 120),
        "count_W": np.random.randint(5, 120),
        "speed_N": np.random.uniform(2, 50),
        "speed_S": np.random.uniform(2, 50),
        "speed_E": np.random.uniform(2, 50),
        "speed_W": np.random.uniform(2, 50),
        "time_of_day": random.choice(["morning", "afternoon", "evening", "night"]),
        "weather": random.choice(["clear", "rainy", "foggy", "cloudy"]),
        "special_event": np.random.choice([0, 1], p=[0.9, 0.1]),
    }
    return data


# -------------------------------
# Predict Congestion Probabilities
# -------------------------------
def predict_congestion(data_dict):
    df = pd.DataFrame([data_dict])

    # Encode categorical columns
    encoded = encoder.transform(df[["time_of_day", "weather"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["time_of_day", "weather"]))

    X = pd.concat([df.drop(columns=["time_of_day", "weather"]).reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

    X = X.reindex(columns=feature_columns, fill_value=0)
    preds = model.predict_proba(X)

    # Collect probabilities for each direction
    directions = ["N", "S", "E", "W"]
    probs = {d: preds[i][0][1] for i, d in enumerate(directions)}
    return probs


# -------------------------------
# Dynamic Signal Decision Logic
# -------------------------------
def decide_phase(probs, base_green=30):
    max_dir = max(probs, key=probs.get)
    confidence = probs[max_dir]

    # Adjust green duration dynamically
    if confidence > 0.75:
        duration = base_green * 1.5
    elif confidence > 0.55:
        duration = base_green * 1.25
    else:
        duration = base_green

    return max_dir, duration, confidence


# -------------------------------
# Main Loop
# -------------------------------
#Testing it over limited loop 
# def run_controller(cycles=10, delay=2):
#     print("🚦 Starting AI Traffic Signal Controller...\n")
#     for cycle in range(1, cycles + 1):
#         data = get_live_data()
#         probs = predict_congestion(data)
#         direction, duration, conf = decide_phase(probs)

#         print(f"🕐 Cycle {cycle} — {datetime.now().strftime('%H:%M:%S')}")
#         print(f"  Live Data: counts={[data[f'count_{d}'] for d in 'NSEW']}, "
#               f"speeds={[round(data[f'speed_{d}'],1) for d in 'NSEW']}")
#         print(f"  Predicted congestion probabilities: {probs}")
#         print(f"  ➤ Selected GREEN: {direction}  |  Duration: {duration:.1f}s  |  Confidence: {conf:.2f}\n")

#         time.sleep(delay)

#     print("✅ Simulation ended successfully.")

#Testing it in infinite mode till user don't stop it manually:
def run_controller(cycles=None , delay=2):
    if cycles == -1 :
        print("🚦 Starting AI Traffic Signal Controller (infinite mode(Press ctrl+c to stop))...\n")
        cycle = 0
        try:
            while True:
                cycle += 1
                data = get_live_data()
                probs = predict_congestion(data)
                direction, duration, conf = decide_phase(probs)

                print(f"🕐 Cycle {cycle}")
                print(f"  Live Data: counts={[data[f'count_{d}'] for d in 'NSEW']}, "
                      f"speeds={[round(data[f'speed_{d}'],1) for d in 'NSEW']}")
                print(f"  Predicted congestion probabilities: {probs}")
                print(f"  ➤ GREEN: {direction} | Duration: {duration:.1f}s | Confidence: {conf:.2f}\n")
                # after print(f"  ➤ GREEN: ...")
                log_entry = {
                    "cycle": cycle,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "counts": [data[f"count_{d}"] for d in "NSEW"],
                    "speeds": [round(data[f"speed_{d}"],1) for d in "NSEW"],
                    "probs": probs,
                    "selected": direction,
                    "duration": round(duration,1),
                    "confidence": round(conf,2)
                }
                pd.DataFrame([log_entry]).to_csv("controller_log.csv", mode="a", header=not os.path.exists("controller_log.csv"), index=False)


                time.sleep(delay)

        except KeyboardInterrupt:
            print(f"\n🛑 Controller stopped after {cycle} cycles, manually by user.\n")
    else :
        print("🚦 Starting AI Traffic Signal Controller...\n")
        for cycle in range(1, cycles + 1):
            data = get_live_data()
            probs = predict_congestion(data)
            direction, duration, conf = decide_phase(probs)

            print(f"🕐 Cycle {cycle} — {datetime.now().strftime('%H:%M:%S')}")
            print(f"  Live Data: counts={[data[f'count_{d}'] for d in 'NSEW']}, "
                  f"speeds={[round(data[f'speed_{d}'],1) for d in 'NSEW']}")
            print(f"  Predicted congestion probabilities: {probs}")
            print(f"  ➤ Selected GREEN: {direction}  |  Duration: {duration:.1f}s  |  Confidence: {conf:.2f}\n")

            time.sleep(delay)

        print("✅ Simulation ended successfully.")


if __name__ == "__main__":
    cycles = int(input("Enter number of cycles (-1 for infinite): ").strip() or -1)
    run_controller(cycles, delay=1)

#next step to add a controller log for future developer use.
