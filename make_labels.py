# make_labels.py
# This creates the fall_labels.csv file.
#
# For recordings containing a fall, you enter:
# 1. When the fall starts
# 2. When the fall ends
#
# The times come from seconds_elapsed in the sensor CSV files.

import os
import pandas as pd

dataRoot = "."
labelFile = "fall_labels.csv"

recordings = []

for root, dirs, files in os.walk(dataRoot):

    if "Accelerometer.csv" in files and "Gyroscope.csv" in files:

        # FIX: previously this used only the folder's own name
        # (os.path.basename(root)), which breaks if two recordings in
        # DIFFERENT activity folders happen to share the same folder
        # name (e.g. two people both naming a take "test1"). Using the
        # path relative to dataRoot instead (e.g. "fall/test1") keeps
        # every recording's identifier unique across all categories.
        recordingName = os.path.relpath(root, dataRoot)

        activityName = os.path.basename(os.path.dirname(root))

        recordings.append([
            recordingName,
            activityName,
            root
        ])

if len(recordings) == 0:

    print("No recordings found.")
    print("Make sure each recording has:")
    print("Accelerometer.csv")
    print("Gyroscope.csv")

else:

    print("\nRecordings found:\n")

    for recording in recordings:
        print(
            recording[0],
            "-",
            recording[1],
            "-",
            recording[2]
        )

    if os.path.exists(labelFile):
        labels = pd.read_csv(labelFile)
    else:
        labels = pd.DataFrame(
            columns=[
                "recording",
                "fall_start",
                "fall_end"
            ]
        )

    print("\nEnter fall times for recordings containing a fall.")
    print("Use the seconds_elapsed column from the sensor file.")
    print("Example: 3.60 to 4.10\n")

    for recordingName, activityName, folder in recordings:

        answer = input(
            recordingName +
            " - does this recording contain a fall? (y/n): "
        )

        if answer.lower() == "y":

            start = input("Fall START time: ")
            end = input("Fall END time: ")

            try:

                start = float(start)
                end = float(end)

                if end <= start:

                    print("The end time must be greater than the start.")
                    continue

                labels = labels[
                    labels["recording"] != recordingName
                ]

                newRow = pd.DataFrame({
                    "recording": [recordingName],
                    "fall_start": [start],
                    "fall_end": [end]
                })

                labels = pd.concat(
                    [labels, newRow],
                    ignore_index=True
                )

            except ValueError:

                print("Please enter numbers.")

        else:

            labels = labels[
                labels["recording"] != recordingName
            ]

    labels.to_csv(
        labelFile,
        index=False
    )

    print("\nSaved:", labelFile)

    print("\nYour labels:")
    print(labels.to_string(index=False))
