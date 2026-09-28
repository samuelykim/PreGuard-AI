# evaluateModel.py
# Evaluate the trained model on the held-out test data.

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report


testData = np.load("testdata.npz")
testX = testData["X"]
testY = testData["y"]

model = tf.keras.models.load_model("fall_model.keras")

scores = model.predict(testX, verbose=0).flatten()


# ---------------------------------------------------------
# DEFAULT THRESHOLD (0.50) RESULTS
# ---------------------------------------------------------

predictions = (scores >= 0.50).astype(int)

print("\nCONFUSION MATRIX (threshold = 0.50)")
print(confusion_matrix(testY, predictions))

print("\nCLASSIFICATION REPORT (threshold = 0.50)")
print(classification_report(
    testY, predictions,
    target_names=["not fall", "fall"],
    zero_division=0
))

print("\nTotal test windows:", len(testY))
print("Actual fall windows:", int(np.sum(testY)))
print("Predicted fall windows:", int(np.sum(predictions)))


# ---------------------------------------------------------
# ADDED: THRESHOLD SWEEP
# ---------------------------------------------------------
# 0.50 is a reasonable DEFAULT starting point, not a guaranteed best
# choice. For an emergency device, missing a real fall (low recall) is
# usually far worse than a false alarm (low precision) -- so it's worth
# checking whether a LOWER threshold (more sensitive, more false alarms)
# gives you meaningfully better recall before locking in 0.50.

if np.sum(testY) == 0:
    print("\nSkipping threshold sweep -- no fall windows in test data yet.")
else:
    from sklearn.metrics import precision_score, recall_score

    print("\nTHRESHOLD SWEEP (how precision/recall trade off):")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10}")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (scores >= threshold).astype(int)
        precision = precision_score(testY, preds, zero_division=0)
        recall = recall_score(testY, preds, zero_division=0)
        print(f"{threshold:>10.2f} {precision:>10.2f} {recall:>10.2f}")

    print("\nPick the threshold that best matches what matters more for "
          "your use case: higher recall (catches more real falls, but "
          "more false alarms) vs higher precision (fewer false alarms, "
          "but might miss some real falls).")
