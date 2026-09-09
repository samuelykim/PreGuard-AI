import pandas as pd
import matplotlib.pyplot as plt



#load both files and see what columns they have
fallFilePath = "fall/fallRecording1.csv"
walkFilePath = "walk/walkRecording1.csv"

fallDataFile = pd.read_csv(fallFilePath)
walkDataFile = pd.read_csv(walkFilePath)

#shows exact column names
print("Columns in the fall recording: ")
print(fallDataFile.columns.tolist())

print("\nColumns in the walking recording: ")
print(walkDataFile.columns.tolist())

print(f"\nFall recording shape: {fallDataFile.shape}") #.shape tells us number of rows and columns (how many data points were got when recording)
print(f"Walking recording shape: {walkDataFile.shape}")

print("\nFirst 5 rows of the fall recording: ") #look at the first few rows so we can see actual number and not names
print(fallDataFile.head())

#pick which column to plot
accelXColumn = "x"
if accelXColumn not in fallDataFile.columns:
    print(f"\n'{accelXColumn}' isnt a real column name in your file. "
          f"Look at the printed column list above and update "
          f"accelXColumn to match exactly (including capitalization).")
else:
    #plot them so we can see the differnce (fall should have a sharp spike where walk will be flat)
    plt.figure(figsize=(10, 4))
    plt.plot(fallDataFile[accelXColumn], label="Fall")
    plt.plot(walkDataFile[accelXColumn], label="Walking")
    plt.legend()
    plt.title("Fall vs Walking - accelerometer data")
    plt.xlabel("Time step")
    plt.ylabel("Acceleration value (check units from app (may be m/s^2))")
    plt.show()
        