import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import optuna
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------
# Plot settings
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("df_cluster.pkl")

# --------------------------------------------------------------
# Encode labels
# --------------------------------------------------------------
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
num_classes = df["label_encoded"].nunique()

# --------------------------------------------------------------
# Feature set (best-performing)
# --------------------------------------------------------------
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [c for c in df.columns if "_temp_" in c]

feature_set = basic_features + square_features + pca_features + time_features

# --------------------------------------------------------------
# Prepare data
# --------------------------------------------------------------
X = df[feature_set]
y = df["label_encoded"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --------------------------------------------------------------
# Optuna objective function
# --------------------------------------------------------------
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 250),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "objective": "multi:softmax",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "tree_method": "hist",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    return acc

# --------------------------------------------------------------
# Run Optuna
# --------------------------------------------------------------
print("\n🔍 Running Optuna optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("\n✅ Best Accuracy:", round(study.best_value * 100, 2), "%")
print("✅ Best Parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# --------------------------------------------------------------
# Train final XGBoost model with best parameters
# --------------------------------------------------------------
best_params = study.best_params
best_params.update({
    "objective": "multi:softmax",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "random_state": 42,
})

xgb_final = XGBClassifier(**best_params)
xgb_final.fit(X_train, y_train)

y_pred = xgb_final.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("\n🎯 Final Validation Accuracy:", round(accuracy * 100, 2), "%")

# --------------------------------------------------------------
# Confusion Matrix (Random Split)
# --------------------------------------------------------------
cm = confusion_matrix(y_val, y_pred)
classes = label_encoder.inverse_transform(np.unique(y_val))

plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix – XGBoost (Optuna)")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Participant-Independent Evaluation (Test = A)
# --------------------------------------------------------------
print("\n👤 Participant-Independent Test (A)")

train_df = df[df["participant"] != "A"]
test_df = df[df["participant"] == "A"]

X_train = train_df[feature_set]
y_train = train_df["label_encoded"]

X_test = test_df[feature_set]
y_test = test_df["label_encoded"]

xgb_final.fit(X_train, y_train)
y_pred = xgb_final.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (Participant A):", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
classes = label_encoder.inverse_transform(np.unique(y_test))

plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix – Participant A (Optuna)")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
