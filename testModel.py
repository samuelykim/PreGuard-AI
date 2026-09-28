# testModel.py
# Test the trained model on a COMPLETE recording.
#
# This shows how the final system will work:
# read sensor data continuously,
# create a sliding window,
# run the model,
# get a fall score.

import os
import numpy as np
import pandas as pd
import tensorflow as tf


scriptFolder = os.path.dirname(os.path.abspath(__file__))

recordingFolder = os.path.join(scriptFolder, "fall", "test31")

sampleRate = 100
windowSeconds = 0.5
stepSeconds = 0.25
windowSize = int(sampleRate * windowSeconds)
stepSize = int(sampleRate * stepSeconds)


model = tf.keras.models.load_model("fall_model.keras")

normalization = np.load("normalization.npz")
mean = normalization["mean"]
std = normalization["std"]


accelFile = os.path.join(recordingFolder, "Accelerometer.csv")
gyroFile = os.path.join(recordingFolder, "Gyroscope.csv")

accel = pd.read_csv(accelFile)
gyro = pd.read_csv(gyroFile)

accel = accel[["seconds_elapsed", "x", "y", "z"]]
gyro = gyro[["seconds_elapsed", "x", "y", "z"]]

accel = accel.rename(columns={"x": "accel_x", "y": "accel_y", "z": "accel_z"})
gyro = gyro.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})

accel = accel.sort_values("seconds_elapsed")
gyro = gyro.sort_values("seconds_elapsed")

data = pd.merge_asof(
    accel, gyro,
    on="seconds_elapsed",
    direction="nearest",
    tolerance=0.02
)
data = data.dropna()


# ---------------------------------------------------------
# BUILD EVERY WINDOW FIRST (FIX: was calling model.predict()
# one window at a time in a loop, which is much slower than
# batching every window into one array and predicting once)
# ---------------------------------------------------------

allWindows = []
windowTimes = []

for start in range(0, len(data) - windowSize + 1, stepSize):

    end = start + windowSize
    window = data.iloc[start:end]

    values = window[
        ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    ].values

    allWindows.append(values)

    startTime = float(window.iloc[0]["seconds_elapsed"])
    endTime = float(window.iloc[-1]["seconds_elapsed"])
    windowTimes.append((startTime, endTime))

allWindows = np.array(allWindows, dtype=np.float32)

# Normalize the whole batch at once, same formula as training.
allWindows = (allWindows - mean) / std


# ---------------------------------------------------------
# ONE BATCHED PREDICTION INSTEAD OF ONE-AT-A-TIME
# ---------------------------------------------------------

scores = model.predict(allWindows, verbose=0).flatten()

results = pd.DataFrame({
    "window_start": [t[0] for t in windowTimes],
    "window_end": [t[1] for t in windowTimes],
    "fall_score_percent": scores * 100
})

results.to_csv("fall_scores.csv", index=False)

print("\nFinished testing the recording.")
print("Saved: fall_scores.csv")

print("\nHighest fall scores:")
highest = results.sort_values("fall_score_percent", ascending=False).head(10)
print(highest.to_string(index=False))
