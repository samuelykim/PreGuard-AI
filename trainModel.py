# trainModel.py
# W5-W6
# Train the baseline fall/not-fall CNN.

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

trainData = np.load("traindata.npz")
valData = np.load("valdata.npz")

trainX = trainData["X"]
trainY = trainData["y"]

valX = valData["X"]
valY = valData["y"]

print("Training shape:", trainX.shape)
print("Validation shape:", valX.shape)


# ---------------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------------
# ADDED: fall windows are a small minority of the data (a fall only
# lasts a second or two inside a much longer recording), and that gap
# gets WORSE as recordings get longer, like your team is planning now.
# Without this, the model can get a deceptively high accuracy score
# just by always guessing "not fall" and ignoring falls entirely.
# Class weighting tells the model to treat mistakes on the rare class
# (fall) as more costly than mistakes on the common class, so it
# actually has to learn to recognize falls rather than ignore them.

numPositive = np.sum(trainY)
numNegative = len(trainY) - numPositive
total = len(trainY)

if numPositive == 0:
    print("\nWARNING: zero fall windows in training data -- class "
          "weighting can't help here. You need at least some fall "
          "windows in the training pile.")
    classWeight = None
else:
    weightForNotFall = total / (2.0 * numNegative)
    weightForFall = total / (2.0 * numPositive)
    classWeight = {0: weightForNotFall, 1: weightForFall}
    print(f"\nClass weights -- not fall: {weightForNotFall:.2f}, "
          f"fall: {weightForFall:.2f}")


# ---------------------------------------------------------
# CREATE CNN (unchanged architecture)
# ---------------------------------------------------------

model = Sequential()

model.add(Conv1D(32, 3, activation="relu",
                  input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation="relu"))
model.add(GlobalAveragePooling1D())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))


# ---------------------------------------------------------
# COMPILE
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# ---------------------------------------------------------
# TRAINING SETTINGS
# ---------------------------------------------------------

earlyStop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
saveBest = ModelCheckpoint("fall_model.keras", monitor="val_loss", save_best_only=True)


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

history = model.fit(
    trainX,
    trainY,
    validation_data=(valX, valY),
    epochs=40,
    batch_size=32,
    class_weight=classWeight,   # ADDED
    callbacks=[earlyStop, saveBest]
)

print("\nBest model saved as fall_model.keras")


# ---------------------------------------------------------
# VALIDATION RESULTS
# ---------------------------------------------------------

results = model.evaluate(valX, valY, verbose=0)

print("\nValidation results:")
print("Loss:", results[0])
print("Accuracy:", results[1])
print("Precision:", results[2])
print("Recall:", results[3])
