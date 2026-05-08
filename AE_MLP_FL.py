import flwr as fl
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

""" this files contains the client code for the federated learning process. 
It defines the IDSClient class that implements the necessary methods for training and evaluating
 the autoencoder model on the client's local data. The client also trains
   a local MLP classifier on the attack data and evaluates the overall IDS performance on the test set.
   Before running this code, ensure that the server is running and that the dataset files are correctly placed and paths updated.
"""

# =========================
# CONFIG
# =========================
TRAIN_NORMAL = ".\\nslkdd_normal.csv"
TRAIN_ATTACK = ".\\nslkdd_attacks.csv"
TEST_FILE    = "C:\\Users\\User\\Desktop\\pfe\\datasets\\NSL-KDD\\KDDTest+.txt"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

LABEL_COLUMN = "label"
NORMAL_LABEL = "normal"

NUMERIC_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "difficulty"
]

# =========================
# LOAD DATA
# =========================
print("[INFO] Chargement des données...")

train_normal_df = pd.read_csv(TRAIN_NORMAL, dtype={col: float for col in NUMERIC_COLUMNS})
train_attack_df = pd.read_csv(TRAIN_ATTACK, dtype={col: float for col in NUMERIC_COLUMNS})
test_df         = pd.read_csv(TEST_FILE, header=None, names=COLUMNS,
                               dtype={col: float for col in NUMERIC_COLUMNS})

# Features
X_train_normal = train_normal_df[NUMERIC_COLUMNS].fillna(0)
X_train_attack = train_attack_df[NUMERIC_COLUMNS].fillna(0)
y_train_attack = train_attack_df[LABEL_COLUMN].astype(str)

X_test = test_df[NUMERIC_COLUMNS].fillna(0)
y_test = test_df[LABEL_COLUMN].astype(str)

print(f"[INFO] Normal train   : {len(X_train_normal):,} lignes")
print(f"[INFO] Attack train   : {len(X_train_attack):,} lignes")
print(f"[INFO] Test           : {len(X_test):,} lignes")

# =========================
# NORMALIZATION
# =========================
scaler = StandardScaler()
X_train_normal_sc = scaler.fit_transform(X_train_normal)
X_train_attack_sc = scaler.transform(X_train_attack)
X_test_sc         = scaler.transform(X_test)

# =========================
# AUTOENCODER
# =========================
def create_autoencoder(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x      = layers.Dense(256, activation='relu')(inputs)
    x      = layers.Dropout(0.2)(x)
    x      = layers.Dense(128,  activation='relu')(x)
    x      = layers.Dropout(0.2)(x)
    latent = layers.Dense(64,  activation='relu')(x)
    x      = layers.Dense(128,  activation='relu')(latent)
    x      = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')

    encoder = Model(inputs, latent)
    return autoencoder, encoder

input_dim = X_train_normal_sc.shape[1]
autoencoder, encoder = create_autoencoder(input_dim)

# =========================
# MLP (ATTACK CLASSIFIER)
# =========================
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    max_iter=300,        
    early_stopping=False, 
    verbose=True
)
# =========================
# UTILS
# =========================
def reconstruction_error(model, data):
    recon = model.predict(data, verbose=0)
    return np.mean((data - recon) ** 2, axis=1)

# =========================
# FEDERATED CLIENT
# ✅ Utilisation de .to_client() — API non dépréciée
# =========================
class IDSClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return autoencoder.get_weights()

    def fit(self, parameters, config):
        autoencoder.set_weights(parameters)
        autoencoder.fit(
            X_train_normal_sc,
            X_train_normal_sc,
            epochs=20,
            batch_size=256,
            verbose=1
        )
        return autoencoder.get_weights(), len(X_train_normal_sc), {}

    def evaluate(self, parameters, config):
        autoencoder.set_weights(parameters)
        loss = autoencoder.evaluate(
            X_train_normal_sc,
            X_train_normal_sc,
            verbose=1
        )
        return float(loss), len(X_train_normal_sc), {}

# =========================
# TRAIN MLP (LOCAL)
# =========================
def train_mlp():
    print("\n[INFO] Entraînement du MLP sur les données d'attaque...")
    latent_attack = encoder.predict(X_train_attack_sc, verbose=0)
    mlp.fit(latent_attack, y_train_attack)
    print("[INFO] MLP entraîné.")

# =========================
# TEST PIPELINE
# ✅ Évaluation en batch (rapide)
# =========================
def evaluate_ids():
    print("\n[INFO] Évaluation du système IDS...")

    # Seuil calculé sur les données normales d'entraînement
    train_errors = reconstruction_error(autoencoder, X_train_normal_sc)

    """threshold    = np.mean(train_errors) + 2 * np.std(train_errors)
    print(f"[INFO] Seuil d'anomalie : {threshold:.6f}")  """

    threshold = np.percentile(train_errors, 90)
    print(f"[INFO] Seuil (percentile 95) : {threshold:.6f}")

    # Erreurs de reconstruction sur le test
    test_errors = reconstruction_error(autoencoder, X_test_sc)

    # Masques normal / anomalie
    normal_mask  = test_errors <  threshold
    anomaly_mask = test_errors >= threshold

    print(f"[INFO] Normal détecté  : {normal_mask.sum():,}")
    print(f"[INFO] Anomalie détectée: {anomaly_mask.sum():,}")

    # Prédictions en batch ✅
    y_pred = np.full(len(X_test_sc), NORMAL_LABEL, dtype=object)

    if anomaly_mask.sum() > 0:
        latent_anomalies  = encoder.predict(X_test_sc[anomaly_mask], verbose=0)
        preds_anomalies   = mlp.predict(latent_anomalies)
        y_pred[anomaly_mask] = preds_anomalies

    print("\n===== IDS PERFORMANCE =====")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix (normal vs attaque) :")
    y_test_binary  = (y_test  != NORMAL_LABEL).astype(int)
    y_pred_binary  = (y_pred  != NORMAL_LABEL).astype(int)
    print(confusion_matrix(y_test_binary, y_pred_binary))

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("[INFO] Démarrage du client Flower...")

    # ✅ API correcte — plus de déprecation warning
    fl.client.start_client(
        server_address="localhost:8090",
        client=IDSClient().to_client()
    )

    # Entraînement local du MLP après FL
    train_mlp()

    # Évaluation complète de l'IDS
    evaluate_ids()
