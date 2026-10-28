"""Train the final classifier on the saved BERTimbau vectors and save it.

Same training as src/bert_eval.py (train split only), so the model saved here is
exactly the one that was evaluated on the test split.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

data = np.load("data/bert_vectors.npz")

model = LogisticRegression(max_iter=1000)
model.fit(data["X_train"], data["y_train"])

print("accuracy on the test split:", model.score(data["X_test"], data["y_test"]))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/classifier.joblib")
print("saved models/classifier.joblib, classes:", model.classes_)
