# prepare_data.py
#
# Combines Accelerometer + Gyroscope data.
# Creates 0.5 second windows with 50% overlap.
#
# IMPORTANT:
# Recordings are split BEFORE windows are created.
# This prevents nearly identical overlapping windows
# from appearing in both training and testing.

import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd

dataRoot = "."
labelFile = "fall_labels.csv"

sampleRate = 100

windowSeconds = 0.5
stepSeconds = 0.25

windowSize = int(sampleRate * windowSeconds)
stepSize = int(sampleRate * stepSeconds)

random.seed(42)


# ---------------------------------------------------------
# LOAD FALL LABELS
# ---------------------------------------------------------

if os.path.exists(labelFile):
    fallLabels = pd.read_csv(labelFile)
else:
    fallLabels = pd.DataFrame(
        columns=["recording", "fall_start", "fall_end"]
    )


# ---------------------------------------------------------
# FIND RECORDINGS
# ---------------------------------------------------------

recordings = []

for root, dirs, files in os.walk(dataRoot):

    if "Accelerometer.csv" in files and "Gyroscope.csv" in files:

        # FIX: matches the same relative-path identifier used in
        # make_labels.py now, so labels actually line up correctly
        # even if two recordings in different folders share a name.
        recordingName = os.path.relpath(root, dataRoot)

        activityName = os.path.basename(os.path.dirname(root))

        recordings.append({
            "name": recordingName,
            "activity": activityName,
            "folder": root
        })


print("Found", len(recordings), "recordings.")

if len(recordings) == 0:
    print("No recordings found.")
    raise SystemExit


# ---------------------------------------------------------
# SPLIT RECORDINGS -- STRATIFIED BY ACTIVITY
# ---------------------------------------------------------
# FIX: previously all recordings were shuffled together as one pile
# before splitting. With few "fall" recordings compared to everything
# else, random chance could easily leave val or test with ZERO fall
# examples, making it impossible to honestly evaluate fall detection.
# Splitting each activity's recordings separately, then combining the
# results, guarantees every activity (including fall) is represented
# proportionally in all three piles.

byActivity = defaultdict(list)
for r in recordings:
    byActivity[r["activity"]].append(r)

trainRecordings = []
valRecordings = []
testRecordings = []

for activity, group in byActivity.items():
    random.shuffle(group)
    n = len(group)
    trainEnd = int(n * 0.70)
    valEnd = int(n * 0.85)

    trainRecordings += group[:trainEnd]
    valRecordings += group[trainEnd:valEnd]
    testRecordings += group[valEnd:]

    print(f"  {activity}: {n} total -> "
          f"{len(group[:trainEnd])} train / "
          f"{len(group[trainEnd:valEnd])} val / "
          f"{len(group[valEnd:])} test")

print("\nRecording split:")
print("Training:", len(trainRecordings))
print("Validation:", len(valRecordings))
print("Testing:", len(testRecordings))

if len(trainRecordings) == 0:
    print("You need more recordings.")
    raise SystemExit


# ---------------------------------------------------------
# FIND FALL TIME
# ---------------------------------------------------------

def getFallInterval(recordingName):

    if len(fallLabels) == 0:
        return None

    matches = fallLabels[fallLabels["recording"] == recordingName]

    if len(matches) == 0:
        return None

    start = float(matches.iloc[0]["fall_start"])
    end = float(matches.iloc[0]["fall_end"])

    return start, end


# ---------------------------------------------------------
# LABEL A WINDOW (center-of-window rule, unchanged)
# ---------------------------------------------------------

def labelWindow(startTime, endTime, fallInterval):

    if fallInterval is None:
        return 0

    fallStart = fallInterval[0]
    fallEnd = fallInterval[1]

    centerTime = (startTime + endTime) / 2

    if centerTime >= fallStart and centerTime <= fallEnd:
        return 1

    return 0


# ---------------------------------------------------------
# LOAD AND COMBINE SENSOR DATA
# ---------------------------------------------------------

def loadRecording(recording):

    accelFile = os.path.join(recording["folder"], "Accelerometer.csv")
    gyroFile = os.path.join(recording["folder"], "Gyroscope.csv")

    accel = pd.read_csv(accelFile)
    gyro = pd.read_csv(gyroFile)

    accel = accel[["seconds_elapsed", "x", "y", "z"]]
    gyro = gyro[["seconds_elapsed", "x", "y", "z"]]

    accel = accel.rename(columns={"x": "accel_x", "y": "accel_y", "z": "accel_z"})
    gyro = gyro.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})

    accel = accel.sort_values("seconds_elapsed")
    gyro = gyro.sort_values("seconds_elapsed")

    combined = pd.merge_asof(
        accel, gyro,
        on="seconds_elapsed",
        direction="nearest",
        tolerance=0.02
    )

    combined = combined.dropna()

    return combined


# ---------------------------------------------------------
# CREATE WINDOWS
# ---------------------------------------------------------

def makeWindows(recordingList):

    allWindows = []
    allLabels = []

    for recording in recordingList:

        print("\nProcessing:", recording["name"])

        data = loadRecording(recording)

        if len(data) < windowSize:
            print("Recording is too short.")
            continue

        fallInterval = getFallInterval(recording["name"])

        numberWindows = 0
        fallWindows = 0

        for start in range(0, len(data) - windowSize + 1, stepSize):

            end = start + windowSize
            window = data.iloc[start:end]

            values = window[
                ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
            ].values

            startTime = float(window.iloc[0]["seconds_elapsed"])
            endTime = float(window.iloc[-1]["seconds_elapsed"])

            label = labelWindow(startTime, endTime, fallInterval)

            allWindows.append(values)
            allLabels.append(label)

            numberWindows += 1
            if label == 1:
                fallWindows += 1

        print("Windows:", numberWindows)
        print("Fall windows:", fallWindows)

    return (
        np.array(allWindows, dtype=np.float32),
        np.array(allLabels, dtype=np.float32)
    )


# ---------------------------------------------------------
# MAKE DATASETS
# ---------------------------------------------------------

trainX, trainY = makeWindows(trainRecordings)
valX, valY = makeWindows(valRecordings)
testX, testY = makeWindows(testRecordings)

print("\nFinal dataset shapes:")
print("Train:", trainX.shape, trainY.shape)
print("Validation:", valX.shape, valY.shape)
print("Test:", testX.shape, testY.shape)


# ---------------------------------------------------------
# NORMALIZE DATA (train-only stats, unchanged -- this was correct)
# ---------------------------------------------------------

mean = np.mean(trainX, axis=(0, 1))
std = np.std(trainX, axis=(0, 1))
std[std == 0] = 1

trainX = (trainX - mean) / std
valX = (valX - mean) / std
testX = (testX - mean) / std

np.savez("normalization.npz", mean=mean, std=std)


# ---------------------------------------------------------
# SAVE DATA
# ---------------------------------------------------------

np.savez("traindata.npz", X=trainX, y=trainY)
np.savez("valdata.npz", X=valX, y=valY)
np.savez("testdata.npz", X=testX, y=testY)

print("\nSaved:")
print("traindata.npz")
print("valdata.npz")
print("testdata.npz")
print("normalization.npz")

print("\nClass counts:")
print("Training falls:", int(np.sum(trainY)))
print("Training not falls:", int(len(trainY) - np.sum(trainY)))
print("Validation falls:", int(np.sum(valY)))
print("Test falls:", int(np.sum(testY)))

if np.sum(valY) == 0 or np.sum(testY) == 0:
    print("\nWARNING: one of your validation/test piles has ZERO fall "
          "windows. Precision/recall for 'fall' can't be meaningfully "
          "measured on that pile until you add more fall recordings.")
