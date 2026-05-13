import os
import copy
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 1. PARAMÈTRES
# =========================================================
TRAIN_FILE = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/KDD2/KDDTrain+.txt"
TEST_FILE  = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/KDD2/KDDTest+.txt"

SAVE_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/Tests/KDD/HFL_AE_MLP_5_CATEGORIES"
os.makedirs(SAVE_FOLDER, exist_ok=True)

NUM_CLIENTS = 5
NUM_EDGE_SERVERS = 2

# Répartition non équitable entre clients.
CLIENT_DATA_PROPORTIONS = [0.40, 0.25, 0.18, 0.12, 0.05]

# Groupes hiérarchiques : clients -> edge servers -> global server.
EDGE_GROUPS = [[0, 1, 2], [3, 4]]

GLOBAL_ROUNDS = 20
LOCAL_EPOCHS_AE = 10
LOCAL_EPOCHS_MLP = 10

BATCH_SIZE = 256
LEARNING_RATE_AE = 1e-3
LEARNING_RATE_MLP = 1e-3

VALID_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 85

USE_RECONSTRUCTION_ERROR_AS_FEATURE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", DEVICE)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# =========================================================
# 2. VALIDATION CONFIG HFL
# =========================================================
def validate_hfl_config():
    if len(CLIENT_DATA_PROPORTIONS) != NUM_CLIENTS:
        raise ValueError("CLIENT_DATA_PROPORTIONS doit avoir la même longueur que NUM_CLIENTS.")
    if not np.isclose(sum(CLIENT_DATA_PROPORTIONS), 1.0):
        raise ValueError("La somme de CLIENT_DATA_PROPORTIONS doit être égale à 1.0.")

    all_clients = sorted([client for group in EDGE_GROUPS for client in group])
    expected_clients = list(range(NUM_CLIENTS))
    if all_clients != expected_clients:
        raise ValueError("EDGE_GROUPS doit contenir chaque client exactement une seule fois.")

    if len(EDGE_GROUPS) != NUM_EDGE_SERVERS:
        raise ValueError("NUM_EDGE_SERVERS doit correspondre au nombre de groupes dans EDGE_GROUPS.")


validate_hfl_config()


# =========================================================
# 3. COLONNES NSL-KDD
# =========================================================
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]


# =========================================================
# 4. REGROUPEMENT DES ATTAQUES EN 5 CATÉGORIES
# =========================================================
DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "mailbomb"
}

PROBE_ATTACKS = {
    "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"
}

R2L_ATTACKS = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
    "snmpguess", "xlock", "xsnoop", "httptunnel", "worm"
}

U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack", "xterm"
}

CATEGORY_ORDER = ["normal", "DoS", "Probe", "R2L", "U2R"]


def map_label_to_category(label):
    label = str(label).strip()

    if label == "normal":
        return "normal"
    if label in DOS_ATTACKS:
        return "DoS"
    if label in PROBE_ATTACKS:
        return "Probe"
    if label in R2L_ATTACKS:
        return "R2L"
    if label in U2R_ATTACKS:
        return "U2R"

    # Sécurité : si un label inconnu apparaît, on le place en R2L par défaut.
    # Tu peux aussi choisir "Unknown" si tu veux une 6e classe.
    print(f"[AVERTISSEMENT] Label inconnu non mappé : {label} -> classé R2L par défaut")
    return "R2L"


# =========================================================
# 5. PRÉTRAITEMENT COMMUN
# =========================================================
def encode_kdd_dataframe(df, feature_columns=None, scaler=None, fit=False):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    original_labels = df["label"].copy().values
    category_labels = np.array([map_label_to_category(lab) for lab in original_labels])
    y_binary = np.array([0 if cat == "normal" else 1 for cat in category_labels], dtype=np.int64)

    df = df.drop(columns=["label", "difficulty"])

    categorical_cols = ["protocol_type", "service", "flag"]
    df = pd.get_dummies(df, columns=categorical_cols)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(0)

    if fit:
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            print("Colonnes constantes supprimées :", len(constant_cols))

        feature_columns = list(df.columns)
        scaler = StandardScaler()
        X = scaler.fit_transform(df).astype(np.float32)
    else:
        missing_cols = [col for col in feature_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in feature_columns]

        df = df.reindex(columns=feature_columns, fill_value=0)

        print("\n===== ALIGNEMENT DATA =====")
        print("Shape après alignement :", df.shape)
        print("Colonnes manquantes ajoutées :", len(missing_cols))
        print("Colonnes extra ignorées      :", len(extra_cols))

        X = scaler.transform(df).astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y_binary, original_labels, category_labels, feature_columns, scaler


def load_train_all_classes(file_path):
    df = pd.read_csv(file_path, names=KDD_COLUMNS)
    df.columns = df.columns.str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    print("\n===== TRAIN BRUT =====")
    print("Shape train brut :", df.shape)
    print("Répartition initiale des labels :")
    print(df["label"].value_counts().head(20))

    X, y_binary, original_labels, category_labels, feature_columns, scaler = encode_kdd_dataframe(df, fit=True)

    print("\n===== TRAIN APRÈS PRÉTRAITEMENT =====")
    print("Shape X :", X.shape)
    print("Nombre de features :", len(feature_columns))
    print("Normaux train  :", np.sum(y_binary == 0))
    print("Attaques train :", np.sum(y_binary == 1))
    print("\nRépartition train par catégorie :")
    print(pd.Series(category_labels).value_counts())

    return X, y_binary, original_labels, category_labels, feature_columns, scaler


def load_test_all_classes(file_path, feature_columns, scaler):
    df = pd.read_csv(file_path, names=KDD_COLUMNS)
    df.columns = df.columns.str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    print("\n===== TEST BRUT =====")
    print("Shape test brut :", df.shape)
    print("Répartition initiale des labels :")
    print(df["label"].value_counts().head(20))

    X, y_binary, original_labels, category_labels, _, _ = encode_kdd_dataframe(
        df,
        feature_columns=feature_columns,
        scaler=scaler,
        fit=False
    )

    print("\nNombre total test      :", len(df))
    print("Normaux réels test    :", np.sum(y_binary == 0))
    print("Attaques réelles test :", np.sum(y_binary == 1))
    print("\nRépartition test par catégorie :")
    print(pd.Series(category_labels).value_counts())

    return X, y_binary, original_labels, category_labels


# =========================================================
# 6. MODÈLE AUTOENCODEUR
# =========================================================
class KDDAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(KDDAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)


# =========================================================
# 7. MODÈLE MLP MULTICLASSE 5 CATÉGORIES
# =========================================================
class MLPFiveCategoryClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPFiveCategoryClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 8. OUTILS HFL
# =========================================================
def split_by_proportions(X, proportions, shuffle=True):
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X[indices]

    n = len(X)
    counts = [int(n * p) for p in proportions]
    counts[-1] = n - sum(counts[:-1])

    splits = []
    start = 0
    for count in counts:
        end = start + count
        splits.append(X[start:end])
        start = end

    return splits


def split_clients_supervised_non_equal(X, y, proportions, shuffle=True):
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

    n = len(X)
    counts = [int(n * p) for p in proportions]
    counts[-1] = n - sum(counts[:-1])

    clients = []
    start = 0
    for count in counts:
        end = start + count
        clients.append((X[start:end], y[start:end]))
        start = end

    return clients


def make_ae_loader(X, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def make_supervised_loader(X, y, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def weighted_average_state_dicts(state_dicts, sizes):
    total_size = sum(sizes)
    if total_size == 0:
        raise ValueError("Impossible d'agréger des modèles avec total_size = 0.")

    avg_weights = copy.deepcopy(state_dicts[0])

    for key in avg_weights.keys():
        if not torch.is_floating_point(avg_weights[key]):
            avg_weights[key] = state_dicts[0][key]
            continue

        avg_weights[key] = torch.zeros_like(avg_weights[key])
        for weights, size in zip(state_dicts, sizes):
            avg_weights[key] += weights[key] * (size / total_size)

    return avg_weights


def hfl_aggregate(local_weights, local_sizes, edge_groups):
    edge_weights = []
    edge_sizes = []

    print("\n===== AGRÉGATION HFL =====")

    for edge_idx, group in enumerate(edge_groups):
        group_weights = [local_weights[i] for i in group]
        group_sizes = [local_sizes[i] for i in group]
        edge_size = sum(group_sizes)

        edge_model_weights = weighted_average_state_dicts(group_weights, group_sizes)
        edge_weights.append(edge_model_weights)
        edge_sizes.append(edge_size)

        print(f"Edge Server {edge_idx + 1} | Clients={ [i + 1 for i in group] } | total_size={edge_size}")

    global_weights = weighted_average_state_dicts(edge_weights, edge_sizes)
    print("Global Server | Edge sizes=", edge_sizes)

    return global_weights, edge_sizes


# =========================================================
# 9. ENTRAÎNEMENT LOCAL AE
# =========================================================
def train_local_ae(global_model, client_X_normal):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    loader = make_ae_loader(client_X_normal, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE_AE)

    losses = []

    for _ in range(LOCAL_EPOCHS_AE):
        running_loss = 0.0

        for batch_x, batch_target in loader:
            batch_x = batch_x.to(DEVICE)
            batch_target = batch_target.to(DEVICE)

            optimizer.zero_grad()
            outputs = local_model(batch_x)
            loss = criterion(outputs, batch_target)

            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        losses.append(running_loss / len(loader.dataset))

    return local_model.state_dict(), len(client_X_normal), float(np.mean(losses))


def compute_reconstruction_errors(model, X):
    model.eval()
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    errors = []

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            batch_errors = torch.mean((batch_x - outputs) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)


def extract_ae_features(model, X):
    model.eval()
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_z = []
    all_errors = []

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            z = model.encode(batch_x)
            x_hat = model(batch_x)
            err = torch.mean((batch_x - x_hat) ** 2, dim=1, keepdim=True)
            all_z.append(z.cpu().numpy())
            all_errors.append(err.cpu().numpy())

    z_all = np.vstack(all_z)
    err_all = np.vstack(all_errors)

    if USE_RECONSTRUCTION_ERROR_AS_FEATURE:
        return np.hstack([z_all, err_all]).astype(np.float32)

    return z_all.astype(np.float32)


# =========================================================
# 10. ENTRAÎNEMENT LOCAL MLP 5 CATÉGORIES
# =========================================================
def train_local_mlp_category(global_model, client_X, client_y, num_classes):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    loader = make_supervised_loader(client_X, client_y, shuffle=True)

    class_counts = np.bincount(client_y, minlength=num_classes)
    total = class_counts.sum()
    weights = total / (num_classes * np.maximum(class_counts, 1))
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE_MLP)

    losses = []

    for _ in range(LOCAL_EPOCHS_MLP):
        running_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = local_model(batch_x)
            loss = criterion(logits, batch_y)

            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        losses.append(running_loss / len(loader.dataset))

    return local_model.state_dict(), len(client_X), float(np.mean(losses))


def predict_mlp_category(model, X):
    model.eval()
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probs = []
    preds = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            p = softmax(logits)
            y = torch.argmax(logits, dim=1)
            probs.append(p.cpu().numpy())
            preds.extend(y.cpu().numpy())

    return np.array(preds), np.vstack(probs)


# =========================================================
# 11. CHARGEMENT DATASET
# =========================================================
print("\nChargement et préparation du train...")
X_all, y_binary_all, train_original_labels, train_category_labels, feature_columns, scaler = load_train_all_classes(TRAIN_FILE)

X_train_full, X_val_full, y_train_binary, y_val_binary, labels_train_original, labels_val_original, labels_train_cat, labels_val_cat = train_test_split(
    X_all,
    y_binary_all,
    train_original_labels,
    train_category_labels,
    test_size=VALID_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_category_labels
)

print("\n===== SPLIT TRAIN/VALIDATION =====")
print("Train shape      :", X_train_full.shape)
print("Validation shape :", X_val_full.shape)
print("Train normaux    :", np.sum(y_train_binary == 0))
print("Train attaques   :", np.sum(y_train_binary == 1))
print("Val normaux      :", np.sum(y_val_binary == 0))
print("Val attaques     :", np.sum(y_val_binary == 1))
print("\nRépartition validation catégories :")
print(pd.Series(labels_val_cat).value_counts())

print("\n===== CONFIGURATION HFL =====")
print("Nombre de clients      :", NUM_CLIENTS)
print("Nombre de Edge servers :", NUM_EDGE_SERVERS)
print("Proportions clients    :", CLIENT_DATA_PROPORTIONS)
print("Groupes Edge           :", [[c + 1 for c in group] for group in EDGE_GROUPS])


# =========================================================
# 12. ÉTAPE 1 : HFL + AE SUR TRAFIC NORMAL
# =========================================================
X_train_normal = X_train_full[y_train_binary == 0]
X_val_normal = X_val_full[y_val_binary == 0]

clients_normal = split_by_proportions(
    X_train_normal,
    CLIENT_DATA_PROPORTIONS,
    shuffle=True
)

print("\n===== CLIENTS AE NORMAL ONLY - RÉPARTITION NON ÉQUITABLE =====")
for i, client_X in enumerate(clients_normal):
    print(f"Client AE {i + 1}: X={client_X.shape}")

input_dim = X_train_full.shape[1]
global_ae = KDDAutoencoder(input_dim).to(DEVICE)

best_ae_val_loss = float("inf")
best_ae_path = os.path.join(SAVE_FOLDER, "best_hfl_autoencoder.pth")

print("\n===== DÉBUT ENTRAÎNEMENT HFL + AE =====")

for round_idx in range(GLOBAL_ROUNDS):
    local_weights = []
    local_sizes = []
    local_losses = []

    print(f"\n--- Round AE {round_idx + 1}/{GLOBAL_ROUNDS} ---")

    for client_idx, client_X in enumerate(clients_normal):
        weights, size, loss = train_local_ae(global_ae, client_X)
        local_weights.append(weights)
        local_sizes.append(size)
        local_losses.append(loss)
        print(f"Client {client_idx + 1} | size={size} | loss={loss:.6f}")

    new_weights, edge_sizes_ae = hfl_aggregate(
        local_weights=local_weights,
        local_sizes=local_sizes,
        edge_groups=EDGE_GROUPS
    )
    global_ae.load_state_dict(new_weights)

    val_errors_normal = compute_reconstruction_errors(global_ae, X_val_normal)
    val_loss = float(np.mean(val_errors_normal))

    print(f"Loss moyenne clients AE       : {np.mean(local_losses):.6f}")
    print(f"Validation AE normal loss     : {val_loss:.6f}")

    if val_loss < best_ae_val_loss:
        best_ae_val_loss = val_loss
        torch.save(global_ae.state_dict(), best_ae_path)
        print(">> Meilleur AE sauvegardé.")

best_ae = KDDAutoencoder(input_dim).to(DEVICE)
best_ae.load_state_dict(torch.load(best_ae_path, map_location=DEVICE))
best_ae.eval()

val_normal_errors = compute_reconstruction_errors(best_ae, X_val_normal)
ae_threshold = np.percentile(val_normal_errors, THRESHOLD_PERCENTILE)

print("\n===== SEUIL AE =====")
print("Erreur moyenne val normal :", np.mean(val_normal_errors))
print("Erreur médiane val normal :", np.median(val_normal_errors))
print(f"Threshold AE ({THRESHOLD_PERCENTILE}e percentile) :", ae_threshold)


# =========================================================
# 13. ÉTAPE 2 : FEATURES AE POUR MLP 5 CATÉGORIES
# Ici, contrairement à la version 22 classes, le MLP apprend aussi normal.
# Classes finales : normal, DoS, Probe, R2L, U2R.
# =========================================================
print("\n===== EXTRACTION FEATURES AE POUR MLP 5 CATÉGORIES =====")

category_encoder = LabelEncoder()
category_encoder.fit(CATEGORY_ORDER)

y_train_category = category_encoder.transform(labels_train_cat)
y_val_category = category_encoder.transform(labels_val_cat)

print("Classes finales apprises :")
print(list(category_encoder.classes_))
print("Nombre de classes finales :", len(category_encoder.classes_))

X_train_mlp = extract_ae_features(best_ae, X_train_full)
X_val_mlp = extract_ae_features(best_ae, X_val_full)

mlp_scaler = StandardScaler()
X_train_mlp = mlp_scaler.fit_transform(X_train_mlp).astype(np.float32)
X_val_mlp = mlp_scaler.transform(X_val_mlp).astype(np.float32)

print("X_train_mlp :", X_train_mlp.shape)
print("X_val_mlp   :", X_val_mlp.shape)


# =========================================================
# 14. ÉTAPE 3 : HFL + MLP 5 CATÉGORIES
# =========================================================
clients_mlp = split_clients_supervised_non_equal(
    X_train_mlp,
    y_train_category,
    CLIENT_DATA_PROPORTIONS,
    shuffle=True
)

print("\n===== CLIENTS MLP 5 CATÉGORIES - RÉPARTITION NON ÉQUITABLE =====")
for i, (client_X, client_y) in enumerate(clients_mlp):
    print(f"\nClient MLP {i + 1}: X={client_X.shape}")
    counts = pd.Series(client_y).value_counts().sort_index()
    for cls_idx, count in counts.items():
        print(f"  {category_encoder.inverse_transform([cls_idx])[0]} : {count}")

mlp_input_dim = X_train_mlp.shape[1]
num_categories = len(category_encoder.classes_)
global_mlp = MLPFiveCategoryClassifier(mlp_input_dim, num_categories).to(DEVICE)

best_val_macro_f1 = -1.0
best_mlp_path = os.path.join(SAVE_FOLDER, "best_hfl_mlp_5_categories.pth")

print("\n===== DÉBUT ENTRAÎNEMENT HFL + MLP 5 CATÉGORIES =====")

for round_idx in range(GLOBAL_ROUNDS):
    local_weights = []
    local_sizes = []
    local_losses = []

    print(f"\n--- Round MLP {round_idx + 1}/{GLOBAL_ROUNDS} ---")

    for client_idx, (client_X, client_y) in enumerate(clients_mlp):
        weights, size, loss = train_local_mlp_category(global_mlp, client_X, client_y, num_categories)
        local_weights.append(weights)
        local_sizes.append(size)
        local_losses.append(loss)
        print(f"Client {client_idx + 1} | size={size} | loss={loss:.6f}")

    new_weights, edge_sizes_mlp = hfl_aggregate(
        local_weights=local_weights,
        local_sizes=local_sizes,
        edge_groups=EDGE_GROUPS
    )
    global_mlp.load_state_dict(new_weights)

    val_pred_category, val_probs_category = predict_mlp_category(global_mlp, X_val_mlp)
    val_acc = accuracy_score(y_val_category, val_pred_category)
    val_macro_f1 = f1_score(y_val_category, val_pred_category, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(y_val_category, val_pred_category, average="weighted", zero_division=0)
    val_attack_binary_pred = (val_pred_category != category_encoder.transform(["normal"])[0]).astype(int)
    val_attack_recall = recall_score(y_val_binary, val_attack_binary_pred, zero_division=0)

    print(f"Loss moyenne clients MLP      : {np.mean(local_losses):.6f}")
    print(f"Validation Accuracy 5 classes : {val_acc:.6f}")
    print(f"Validation Macro-F1 5 classes : {val_macro_f1:.6f}")
    print(f"Validation Weighted-F1        : {val_weighted_f1:.6f}")
    print(f"Validation Recall Attack      : {val_attack_recall:.6f}")

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        torch.save(global_mlp.state_dict(), best_mlp_path)
        print(">> Meilleur MLP 5 catégories sauvegardé.")

best_mlp = MLPFiveCategoryClassifier(mlp_input_dim, num_categories).to(DEVICE)
best_mlp.load_state_dict(torch.load(best_mlp_path, map_location=DEVICE))
best_mlp.eval()


# =========================================================
# 15. TEST FINAL ALL-PASS : TOUS LES ÉCHANTILLONS PASSENT AU MLP
# Input -> AE -> features AE + reconstruction error -> MLP -> 5 catégories
# =========================================================
print("\nChargement et préparation du test...")
X_test, y_test_binary, original_labels, test_category_labels = load_test_all_classes(TEST_FILE, feature_columns, scaler)

test_errors = compute_reconstruction_errors(best_ae, X_test)
ae_pred_binary = (test_errors > ae_threshold).astype(int)

X_test_mlp = extract_ae_features(best_ae, X_test)
X_test_mlp = mlp_scaler.transform(X_test_mlp).astype(np.float32)

pred_category_encoded, pred_category_probs = predict_mlp_category(best_mlp, X_test_mlp)
pred_category_labels = category_encoder.inverse_transform(pred_category_encoded)
mlp_confidence = np.max(pred_category_probs, axis=1)

normal_class_id = category_encoder.transform(["normal"])[0]
y_pred_binary = np.where(pred_category_encoded == normal_class_id, 0, 1)
y_test_category = category_encoder.transform(test_category_labels)

print("\n===== PIPELINE FINAL 5 CATÉGORIES ALL-PASS =====")
print("Normaux réels                       :", np.sum(y_test_binary == 0))
print("Attaques réelles                    :", np.sum(y_test_binary == 1))
print("Prédictions AE normal               :", np.sum(ae_pred_binary == 0))
print("Prédictions AE anomalie             :", np.sum(ae_pred_binary == 1))
print("Échantillons envoyés au MLP         :", len(X_test))
print("Normaux prédits finaux par MLP      :", np.sum(y_pred_binary == 0))
print("Attaques prédites finales par MLP   :", np.sum(y_pred_binary == 1))


# =========================================================
# 16. MÉTRIQUES BINAIRES : NORMAL VS ATTAQUE
# =========================================================
acc = accuracy_score(y_test_binary, y_pred_binary)
prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

cm = confusion_matrix(y_test_binary, y_pred_binary, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

try:
    roc_auc_ae = roc_auc_score(y_test_binary, test_errors)
except Exception:
    roc_auc_ae = np.nan

try:
    attack_scores = 1.0 - pred_category_probs[:, normal_class_id]
    roc_auc_mlp = roc_auc_score(y_test_binary, attack_scores)
except Exception:
    roc_auc_mlp = np.nan

print("\n===== RÉSULTATS BINAIRES TEST : AE -> MLP 5 CATÉGORIES ALL-PASS =====")
print("Accuracy              :", acc)
print("Precision             :", prec)
print("Recall / DetectionRate:", rec)
print("F1-score              :", f1)
print("Specificity           :", specificity)
print("False Positive Rate   :", fpr)
print("False Negative Rate   :", fnr)
print("ROC-AUC AE error      :", roc_auc_ae)
print("ROC-AUC MLP attack    :", roc_auc_mlp)
print("Threshold AE analysé  :", ae_threshold)

print("\n===== MATRICE DE CONFUSION BINAIRE =====")
print(cm)

print("\n===== CLASSIFICATION REPORT BINAIRE =====")
print(classification_report(
    y_test_binary,
    y_pred_binary,
    labels=[0, 1],
    target_names=["Normal", "Attack"],
    digits=4,
    zero_division=0
))


# =========================================================
# 17. MÉTRIQUES 5 CATÉGORIES
# =========================================================
cat_acc = accuracy_score(y_test_category, pred_category_encoded)
cat_macro_f1 = f1_score(y_test_category, pred_category_encoded, average="macro", zero_division=0)
cat_weighted_f1 = f1_score(y_test_category, pred_category_encoded, average="weighted", zero_division=0)

print("\n===== ÉVALUATION 5 CATÉGORIES =====")
print("Accuracy 5 catégories :", cat_acc)
print("Macro-F1 5 catégories :", cat_macro_f1)
print("Weighted-F1 5 catégories :", cat_weighted_f1)

print("\n===== MATRICE DE CONFUSION 5 CATÉGORIES =====")
cat_labels_ids = list(range(len(category_encoder.classes_)))
cat_cm = confusion_matrix(y_test_category, pred_category_encoded, labels=cat_labels_ids)
print(pd.DataFrame(cat_cm, index=category_encoder.classes_, columns=category_encoder.classes_))

print("\n===== CLASSIFICATION REPORT 5 CATÉGORIES =====")
print(classification_report(
    y_test_category,
    pred_category_encoded,
    labels=cat_labels_ids,
    target_names=category_encoder.classes_,
    digits=4,
    zero_division=0
))

print("\n===== ERREURS DE RECONSTRUCTION TEST =====")
print(pd.Series(test_errors).describe())

print("\n===== CONFIANCE MLP SUR TOUS LES ÉCHANTILLONS =====")
print(pd.Series(mlp_confidence).describe())


# =========================================================
# 18. SAUVEGARDE
# =========================================================
results_df = pd.DataFrame({
    "label_original": original_labels,
    "true_category": test_category_labels,
    "pred_category": pred_category_labels,
    "y_true_binary": y_test_binary,
    "ae_pred_binary": ae_pred_binary,
    "final_pred_binary": y_pred_binary,
    "reconstruction_error": test_errors,
    "mlp_confidence": mlp_confidence,
    "mlp_attack_score": attack_scores if 'attack_scores' in locals() else np.nan
})

results_path = os.path.join(SAVE_FOLDER, "results_test_hfl_ae_mlp_5_categories.csv")
results_df.to_csv(results_path, index=False)

metrics = {
    "binary_accuracy": acc,
    "binary_precision": prec,
    "binary_recall_detection_rate": rec,
    "binary_f1_score": f1,
    "binary_specificity": specificity,
    "binary_fpr": fpr,
    "binary_fnr": fnr,
    "binary_roc_auc_ae_error": roc_auc_ae,
    "binary_roc_auc_mlp_attack": roc_auc_mlp,
    "category_accuracy": cat_acc,
    "category_macro_f1": cat_macro_f1,
    "category_weighted_f1": cat_weighted_f1,
    "ae_threshold": ae_threshold,
    "best_ae_val_loss": best_ae_val_loss,
    "best_mlp_val_macro_f1": best_val_macro_f1,
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}

metadata = {
    "architecture": "HFL: clients -> edge servers -> global server ; Input -> Autoencoder -> AE features + reconstruction_error -> MLP -> 5 categories",
    "final_classes": list(category_encoder.classes_),
    "category_mapping": {
        "normal": ["normal"],
        "DoS": sorted(list(DOS_ATTACKS)),
        "Probe": sorted(list(PROBE_ATTACKS)),
        "R2L": sorted(list(R2L_ATTACKS)),
        "U2R": sorted(list(U2R_ATTACKS))
    },
    "input_dim": input_dim,
    "mlp_input_dim": mlp_input_dim,
    "num_categories": num_categories,
    "num_clients": NUM_CLIENTS,
    "num_edge_servers": NUM_EDGE_SERVERS,
    "edge_groups": [[c + 1 for c in group] for group in EDGE_GROUPS],
    "client_data_proportions": CLIENT_DATA_PROPORTIONS,
    "global_rounds": GLOBAL_ROUNDS,
    "local_epochs_ae": LOCAL_EPOCHS_AE,
    "local_epochs_mlp": LOCAL_EPOCHS_MLP,
    "batch_size": BATCH_SIZE,
    "learning_rate_ae": LEARNING_RATE_AE,
    "learning_rate_mlp": LEARNING_RATE_MLP,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "use_reconstruction_error_as_feature": USE_RECONSTRUCTION_ERROR_AS_FEATURE
}

joblib.dump(scaler, os.path.join(SAVE_FOLDER, "scaler_kdd.pkl"))
joblib.dump(mlp_scaler, os.path.join(SAVE_FOLDER, "mlp_feature_scaler.pkl"))
joblib.dump(feature_columns, os.path.join(SAVE_FOLDER, "feature_columns_kdd.pkl"))
joblib.dump(ae_threshold, os.path.join(SAVE_FOLDER, "ae_threshold.pkl"))
joblib.dump(category_encoder, os.path.join(SAVE_FOLDER, "category_label_encoder.pkl"))
joblib.dump(metrics, os.path.join(SAVE_FOLDER, "metrics_hfl_ae_mlp_5_categories.pkl"))
joblib.dump(metadata, os.path.join(SAVE_FOLDER, "metadata_hfl_ae_mlp_5_categories.pkl"))

pd.DataFrame(cat_cm, index=category_encoder.classes_, columns=category_encoder.classes_).to_csv(
    os.path.join(SAVE_FOLDER, "confusion_matrix_5_categories.csv")
)

torch.save(best_ae.state_dict(), os.path.join(SAVE_FOLDER, "final_hfl_autoencoder.pth"))
torch.save(best_mlp.state_dict(), os.path.join(SAVE_FOLDER, "final_hfl_mlp_5_categories.pth"))

print("\n===== SAUVEGARDE TERMINÉE =====")
print("Dossier résultats :", SAVE_FOLDER)
print("Fichier résultats :", results_path)
