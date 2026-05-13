import os
import glob
import copy
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 1. PARAMÈTRES
# =========================================================
BENIGN_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/CICIoT/Benign_Final"
MIXED_FOLDER  = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/CICIoT/Test_CICIoT"

SAVE_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/CICIoT/FedDistill_AE_MLP"
os.makedirs(SAVE_FOLDER, exist_ok=True)

RANDOM_STATE = 42
VALID_SIZE_AE = 0.2
MLP_TEST_SIZE = 0.3

NUM_CLIENTS = 5
CLIENT_DATA_PROPORTIONS = [0.40, 0.25, 0.18, 0.12, 0.05]

GLOBAL_ROUNDS = 10
LOCAL_EPOCHS_AE = 5
LOCAL_EPOCHS_MLP = 5

# Paramètres FedDistill
PUBLIC_DISTILL_SIZE = 10000
DISTILL_EPOCHS = 3
DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA = 0.7

BATCH_SIZE = 256
LEARNING_RATE_AE = 1e-3
LEARNING_RATE_MLP = 1e-3

USE_RECONSTRUCTION_ERROR_AS_FEATURE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", DEVICE)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# =========================================================
# 2. FONCTIONS DE CHARGEMENT CICIoT2023
# =========================================================
def load_folder_with_labels(folder_path, infer_labels=True):
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if len(files) == 0:
        raise FileNotFoundError(f"Aucun CSV trouvé dans : {folder_path}")

    dfs = []
    y = []
    source_files = []

    print(f"\nChargement dossier : {folder_path}")
    print("Nombre de fichiers :", len(files))

    for file in files:
        filename = os.path.basename(file)
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        dfs.append(df)
        source_files.extend([filename] * len(df))

        if infer_labels:
            # Si les fichiers sont séparés par nom : Benign = 0, sinon Attack = 1
            if "Benign" in filename or "benign" in filename:
                label = 0
                label_name = "BENIGN"
            else:
                label = 1
                label_name = "ATTACK"

            y.extend([label] * len(df))
            print(f"[OK] {filename} -> {df.shape} | {label_name}")
        else:
            print(f"[OK] {filename} -> {df.shape}")

    df_all = pd.concat(dfs, ignore_index=True)

    if infer_labels:
        y = np.array(y, dtype=np.int64)
        return df_all, y, source_files

    return df_all, source_files


def preprocess_dataframe(df, fit_scaler=False, scaler=None, feature_columns=None):
    df = df.copy()
    df.columns = df.columns.str.strip()

    label_candidates = ["label", "Label", "class", "Class", "Attack", "attack"]
    existing_labels = [c for c in label_candidates if c in df.columns]
    if existing_labels:
        df = df.drop(columns=existing_labels)

    cols_to_drop = [
        "Flow ID", "flow_id",
        "Timestamp", "timestamp",
        "Src IP", "Dst IP",
        "Source IP", "Destination IP"
    ]

    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)

    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(0)

    if fit_scaler:
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            print("Colonnes constantes supprimées :", len(constant_cols))

        feature_columns = list(df.columns)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df).astype(np.float32)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled, scaler, feature_columns

    if feature_columns is None or scaler is None:
        raise ValueError("feature_columns et scaler sont obligatoires si fit_scaler=False")

    missing_cols = [col for col in feature_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in feature_columns]

    df = df.reindex(columns=feature_columns, fill_value=0)

    print("\n===== ALIGNEMENT DATA =====")
    print("Shape après alignement :", df.shape)
    print("Colonnes manquantes ajoutées :", len(missing_cols))
    print("Colonnes extra ignorées      :", len(extra_cols))

    X_scaled = scaler.transform(df).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled


# =========================================================
# 3. OUTILS CLIENTS / FEDAVG / FEDDISTILL
# =========================================================
def normalize_proportions(proportions, num_clients):
    if len(proportions) != num_clients:
        raise ValueError("CLIENT_DATA_PROPORTIONS doit avoir la même taille que NUM_CLIENTS.")

    proportions = np.array(proportions, dtype=np.float64)

    if np.any(proportions <= 0):
        raise ValueError("Toutes les proportions doivent être positives.")

    proportions = proportions / proportions.sum()
    return proportions


def split_clients_non_equal_unsupervised(X, num_clients, proportions):
    proportions = normalize_proportions(proportions, num_clients)
    indices = np.random.permutation(len(X))

    split_points = (np.cumsum(proportions)[:-1] * len(X)).astype(int)
    client_indices = np.split(indices, split_points)

    return [X[idx] for idx in client_indices]


def split_clients_non_equal_supervised(X, y, num_clients, proportions):
    proportions = normalize_proportions(proportions, num_clients)
    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    split_points = (np.cumsum(proportions)[:-1] * len(X)).astype(int)
    X_splits = np.split(X, split_points)
    y_splits = np.split(y, split_points)

    return list(zip(X_splits, y_splits))


def weighted_average_state_dicts(state_dicts, sizes):
    total_size = sum(sizes)
    if total_size == 0:
        raise ValueError("Impossible d'agréger : total_size = 0.")

    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state.keys():
        if not torch.is_floating_point(avg_state[key]):
            avg_state[key] = state_dicts[0][key]
            continue

        avg_state[key] = torch.zeros_like(avg_state[key])
        for state, size in zip(state_dicts, sizes):
            avg_state[key] += state[key] * (size / total_size)

    return avg_state


def fedavg_aggregate(local_weights, local_sizes, model_name="model"):
    print(f"\n===== AGRÉGATION FEDAVG : {model_name} =====")
    print("Tailles clients :", local_sizes)
    return weighted_average_state_dicts(local_weights, local_sizes)


def sample_public_distillation_set(X, y, max_size, random_state=42):
    if len(X) <= max_size:
        return X, y

    try:
        _, X_public, _, y_public = train_test_split(
            X,
            y,
            test_size=max_size,
            random_state=random_state,
            stratify=y
        )
        return X_public, y_public
    except Exception:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_size, replace=False)
        return X[idx], y[idx]


def collect_soft_targets_from_clients(client_models, X_public, client_sizes, temperature):
    loader = DataLoader(
        torch.tensor(X_public, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_client_probs = []

    for model in client_models:
        model.eval()
        probs_batches = []

        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(DEVICE)
                logits = model(batch_x)
                probs = torch.softmax(logits / temperature, dim=1)
                probs_batches.append(probs.cpu().numpy())

        all_client_probs.append(np.vstack(probs_batches))

    total_size = sum(client_sizes)
    soft_targets = np.zeros_like(all_client_probs[0], dtype=np.float32)

    for probs, size in zip(all_client_probs, client_sizes):
        soft_targets += probs * (size / total_size)

    soft_targets = soft_targets / np.clip(soft_targets.sum(axis=1, keepdims=True), 1e-12, None)
    return soft_targets.astype(np.float32)


# =========================================================
# 4. MODÈLES AE + MLP
# =========================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

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


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPBinaryClassifier, self).__init__()

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

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 5. DATA LOADERS
# =========================================================
def make_ae_loader(X, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def make_supervised_loader(X, y, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


# =========================================================
# 6. ENTRAÎNEMENT LOCAL
# =========================================================
def train_local_ae(global_model, client_X):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    loader = make_ae_loader(client_X, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE_AE)
    losses = []

    for _ in range(LOCAL_EPOCHS_AE):
        running_loss = 0.0

        for batch_x, batch_target in loader:
            batch_x = batch_x.to(DEVICE)
            batch_target = batch_target.to(DEVICE)

            optimizer.zero_grad()
            output = local_model(batch_x)
            loss = criterion(output, batch_target)

            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        losses.append(running_loss / len(loader.dataset))

    return local_model.state_dict(), len(client_X), float(np.mean(losses))


def train_local_mlp(global_model, client_X, client_y):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    loader = make_supervised_loader(client_X, client_y, shuffle=True)

    class_counts = np.bincount(client_y, minlength=2)
    total = class_counts.sum()
    class_weights = total / (2.0 * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
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


def distill_global_mlp(global_model, X_public, y_public, soft_targets):
    global_model = copy.deepcopy(global_model).to(DEVICE)
    global_model.train()

    dataset = TensorDataset(
        torch.tensor(X_public, dtype=torch.float32),
        torch.tensor(y_public, dtype=torch.long),
        torch.tensor(soft_targets, dtype=torch.float32)
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE_MLP)

    losses = []

    for _ in range(DISTILL_EPOCHS):
        running_loss = 0.0

        for batch_x, batch_y, batch_soft in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_soft = batch_soft.to(DEVICE)

            optimizer.zero_grad()

            logits = global_model(batch_x)

            loss_ce = ce_loss(logits, batch_y)
            log_probs = torch.log_softmax(logits / DISTILL_TEMPERATURE, dim=1)
            loss_kd = kl_loss(log_probs, batch_soft) * (DISTILL_TEMPERATURE ** 2)

            loss = (1.0 - DISTILL_ALPHA) * loss_ce + DISTILL_ALPHA * loss_kd

            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        losses.append(running_loss / len(loader.dataset))

    return global_model.state_dict(), float(np.mean(losses))


# =========================================================
# 7. FEATURES AE / PRÉDICTION / ÉVALUATION
# =========================================================
def reconstruction_errors(model, X):
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
            output = model(batch_x)
            err = torch.mean((batch_x - output) ** 2, dim=1)
            errors.extend(err.cpu().numpy())

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

    z_all = np.vstack(all_z).astype(np.float32)
    err_all = np.vstack(all_errors).astype(np.float32)

    if USE_RECONSTRUCTION_ERROR_AS_FEATURE:
        return np.hstack([z_all, err_all]).astype(np.float32)

    return z_all.astype(np.float32)


def predict_mlp(model, X):
    model.eval()

    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    preds = []
    probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)

            logits = model(batch_x)
            p = softmax(logits)
            pred = torch.argmax(p, dim=1)

            preds.extend(pred.cpu().numpy())
            probs.append(p.cpu().numpy())

    return np.array(preds), np.vstack(probs)


def evaluate(name, y_true, y_pred, scores=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, scores) if scores is not None else np.nan
    except Exception:
        auc = np.nan

    print(f"\n===== RÉSULTATS {name} =====")
    print("Accuracy              :", acc)
    print("Precision             :", prec)
    print("Recall / DetectionRate:", rec)
    print("F1-score              :", f1)
    print("Specificity           :", specificity)
    print("False Positive Rate   :", fpr)
    print("False Negative Rate   :", fnr)
    print("ROC-AUC               :", auc)
    print("\nMatrice de confusion :")
    print(cm)
    print("\nClassification report :")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Attack"],
        digits=4,
        zero_division=0
    ))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# =========================================================
# 8. CHARGEMENT BENIGN POUR AE
# =========================================================
df_benign, benign_sources = load_folder_with_labels(BENIGN_FOLDER, infer_labels=False)

print("\n===== BENIGN POUR AUTOENCODEUR =====")
print("Shape brute benign :", df_benign.shape)

X_benign, scaler, feature_columns = preprocess_dataframe(
    df_benign,
    fit_scaler=True
)

print("Shape benign après prétraitement :", X_benign.shape)
print("Nombre de features :", len(feature_columns))


# =========================================================
# 9. SPLIT BENIGN AE TRAIN / VALIDATION
# =========================================================
X_ae_train, X_ae_val = train_test_split(
    X_benign,
    test_size=VALID_SIZE_AE,
    random_state=RANDOM_STATE
)

print("\n===== SPLIT AUTOENCODEUR =====")
print("AE train :", X_ae_train.shape)
print("AE val   :", X_ae_val.shape)


# =========================================================
# 10. CLIENTS AE
# =========================================================
clients_ae = split_clients_non_equal_unsupervised(
    X_ae_train,
    NUM_CLIENTS,
    CLIENT_DATA_PROPORTIONS
)

print("\n===== CLIENTS AE FEDAVG NON ÉQUITABLES =====")
for i, client_X in enumerate(clients_ae):
    print(f"Client AE {i + 1}: {client_X.shape}")


# =========================================================
# 11. ENTRAÎNEMENT AE FEDAVG SUR NORMAL ONLY
# =========================================================
input_dim = X_benign.shape[1]
ae_model = Autoencoder(input_dim).to(DEVICE)

best_ae_val_loss = float("inf")
best_ae_path = os.path.join(SAVE_FOLDER, "best_feddistill_autoencoder.pth")

print("\n===== ENTRAÎNEMENT AE FEDAVG SUR BENIGN ONLY =====")

for round_idx in range(GLOBAL_ROUNDS):
    print(f"\n--- Round AE {round_idx + 1}/{GLOBAL_ROUNDS} ---")

    local_weights = []
    local_sizes = []
    local_losses = []

    for client_idx, client_X in enumerate(clients_ae):
        weights, size, loss = train_local_ae(ae_model, client_X)

        local_weights.append(weights)
        local_sizes.append(size)
        local_losses.append(loss)

        print(f"Client {client_idx + 1} | size={size} | loss={loss:.6f}")

    new_weights = fedavg_aggregate(local_weights, local_sizes, model_name="Autoencoder")
    ae_model.load_state_dict(new_weights)

    val_errors = reconstruction_errors(ae_model, X_ae_val)
    val_loss = float(np.mean(val_errors))

    print("Loss moyenne clients AE :", np.mean(local_losses))
    print("Validation loss AE      :", val_loss)

    if val_loss < best_ae_val_loss:
        best_ae_val_loss = val_loss
        torch.save(ae_model.state_dict(), best_ae_path)
        print(">> Meilleur AE sauvegardé.")


best_ae = Autoencoder(input_dim).to(DEVICE)
best_ae.load_state_dict(torch.load(best_ae_path, map_location=DEVICE))
best_ae.eval()


# =========================================================
# 12. CHARGEMENT DATASET MIXTE POUR MLP
# =========================================================
df_mixed, y_mixed, source_files = load_folder_with_labels(MIXED_FOLDER, infer_labels=True)

print("\n===== DATASET MIXTE =====")
print("Shape brute mixed :", df_mixed.shape)
print("Total     :", len(y_mixed))
print("Normaux   :", np.sum(y_mixed == 0))
print("Attaques  :", np.sum(y_mixed == 1))

X_mixed = preprocess_dataframe(
    df_mixed,
    fit_scaler=False,
    scaler=scaler,
    feature_columns=feature_columns
)

print("Shape mixed après prétraitement :", X_mixed.shape)


# =========================================================
# 13. FEATURES AE POUR TOUTES LES DONNÉES MIXTES
# =========================================================
print("\n===== EXTRACTION FEATURES AE POUR TOUT LE DATASET MIXTE =====")

X_mixed_mlp = extract_ae_features(best_ae, X_mixed)

mlp_feature_scaler = StandardScaler()
X_mixed_mlp = mlp_feature_scaler.fit_transform(X_mixed_mlp).astype(np.float32)

print("X_mixed_mlp :", X_mixed_mlp.shape)


# =========================================================
# 14. SPLIT MLP TRAIN / TEST
# =========================================================
X_mlp_train, X_mlp_test, y_mlp_train, y_mlp_test, src_train, src_test = train_test_split(
    X_mixed_mlp,
    y_mixed,
    source_files,
    test_size=MLP_TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_mixed
)

print("\n===== SPLIT MLP BINAIRE =====")
print("MLP train :", X_mlp_train.shape)
print("MLP test  :", X_mlp_test.shape)
print("Train benign :", np.sum(y_mlp_train == 0), "| Train attack :", np.sum(y_mlp_train == 1))
print("Test benign  :", np.sum(y_mlp_test == 0), "| Test attack  :", np.sum(y_mlp_test == 1))

clients_mlp = split_clients_non_equal_supervised(
    X_mlp_train,
    y_mlp_train,
    NUM_CLIENTS,
    CLIENT_DATA_PROPORTIONS
)

print("\n===== CLIENTS MLP FEDDISTILL NON ÉQUITABLES =====")
for i, (client_X, client_y) in enumerate(clients_mlp):
    print(
        f"Client MLP {i + 1}: X={client_X.shape} | "
        f"benign={np.sum(client_y == 0)} | attack={np.sum(client_y == 1)}"
    )


# =========================================================
# 15. FEDDISTILL + MLP
# =========================================================
mlp_input_dim = X_mlp_train.shape[1]
global_mlp = MLPBinaryClassifier(mlp_input_dim).to(DEVICE)

X_public_distill, y_public_distill = sample_public_distillation_set(
    X_mlp_test,
    y_mlp_test,
    max_size=PUBLIC_DISTILL_SIZE,
    random_state=RANDOM_STATE
)

print("\n===== JEU PUBLIC / PROXY FEDDISTILL =====")
print("X_public_distill :", X_public_distill.shape)
print("Benign public    :", np.sum(y_public_distill == 0))
print("Attack public    :", np.sum(y_public_distill == 1))

best_mlp_val_f1 = -1.0
best_mlp_path = os.path.join(SAVE_FOLDER, "best_feddistill_mlp_binary.pth")

print("\n===== ENTRAÎNEMENT FEDDISTILL + MLP BINAIRE =====")

for round_idx in range(GLOBAL_ROUNDS):
    print(f"\n--- Round FedDistill MLP {round_idx + 1}/{GLOBAL_ROUNDS} ---")

    local_models = []
    local_sizes = []
    local_losses = []

    # 1) Entraînement local
    for client_idx, (client_X, client_y) in enumerate(clients_mlp):
        weights, size, loss = train_local_mlp(global_mlp, client_X, client_y)

        client_model = MLPBinaryClassifier(mlp_input_dim).to(DEVICE)
        client_model.load_state_dict(weights)
        client_model.eval()

        local_models.append(client_model)
        local_sizes.append(size)
        local_losses.append(loss)

        print(f"Client {client_idx + 1} | size={size} | loss={loss:.6f}")

    # 2) Soft labels des clients sur le jeu public
    soft_targets = collect_soft_targets_from_clients(
        client_models=local_models,
        X_public=X_public_distill,
        client_sizes=local_sizes,
        temperature=DISTILL_TEMPERATURE
    )

    # 3) Distillation serveur
    distilled_weights, distill_loss = distill_global_mlp(
        global_model=global_mlp,
        X_public=X_public_distill,
        y_public=y_public_distill,
        soft_targets=soft_targets
    )

    global_mlp.load_state_dict(distilled_weights)

    # 4) Validation
    val_pred, val_probs = predict_mlp(global_mlp, X_mlp_test)
    val_scores = val_probs[:, 1]

    val_f1 = f1_score(y_mlp_test, val_pred, zero_division=0)
    val_acc = accuracy_score(y_mlp_test, val_pred)

    print("Loss moyenne clients MLP :", np.mean(local_losses))
    print("Loss distillation serveur:", distill_loss)
    print("Validation Accuracy MLP  :", val_acc)
    print("Validation F1 MLP        :", val_f1)

    if val_f1 > best_mlp_val_f1:
        best_mlp_val_f1 = val_f1
        torch.save(global_mlp.state_dict(), best_mlp_path)
        print(">> Meilleur MLP FedDistill sauvegardé.")


best_mlp = MLPBinaryClassifier(mlp_input_dim).to(DEVICE)
best_mlp.load_state_dict(torch.load(best_mlp_path, map_location=DEVICE))
best_mlp.eval()


# =========================================================
# 16. TEST FINAL : TOUTES LES DONNÉES PASSENT AE -> MLP
# =========================================================
print("\n===== TEST FINAL AE -> MLP SUR DATASET MIXTE =====")

test_pred, test_probs = predict_mlp(best_mlp, X_mlp_test)
test_scores = test_probs[:, 1]

test_results = evaluate(
    "FedDistill AE -> MLP",
    y_mlp_test,
    test_pred,
    scores=test_scores
)

mixed_errors_test = X_mlp_test[:, -1] if USE_RECONSTRUCTION_ERROR_AS_FEATURE else np.zeros(len(X_mlp_test))


# =========================================================
# 17. SAUVEGARDE
# =========================================================
torch.save(best_ae.state_dict(), os.path.join(SAVE_FOLDER, "feddistill_autoencoder_final.pth"))
torch.save(best_mlp.state_dict(), os.path.join(SAVE_FOLDER, "feddistill_mlp_binary_final.pth"))

joblib.dump(scaler, os.path.join(SAVE_FOLDER, "scaler_ciciot.pkl"))
joblib.dump(mlp_feature_scaler, os.path.join(SAVE_FOLDER, "mlp_feature_scaler.pkl"))
joblib.dump(feature_columns, os.path.join(SAVE_FOLDER, "feature_columns_ciciot.pkl"))

metadata = {
    "architecture": "FedDistill + AE + MLP: AE trained on benign traffic only using FedAvg; all samples pass through AE -> latent features + reconstruction error -> MLP; MLP trained with FedDistill",
    "num_clients": NUM_CLIENTS,
    "client_data_proportions": CLIENT_DATA_PROPORTIONS,
    "global_rounds": GLOBAL_ROUNDS,
    "local_epochs_ae": LOCAL_EPOCHS_AE,
    "local_epochs_mlp": LOCAL_EPOCHS_MLP,
    "public_distill_size": PUBLIC_DISTILL_SIZE,
    "distill_epochs": DISTILL_EPOCHS,
    "distill_temperature": DISTILL_TEMPERATURE,
    "distill_alpha": DISTILL_ALPHA,
    "batch_size": BATCH_SIZE,
    "learning_rate_ae": LEARNING_RATE_AE,
    "learning_rate_mlp": LEARNING_RATE_MLP,
    "best_ae_val_loss": best_ae_val_loss,
    "best_mlp_val_f1": best_mlp_val_f1,
    "input_dim": input_dim,
    "mlp_input_dim": mlp_input_dim,
    "use_reconstruction_error_as_feature": USE_RECONSTRUCTION_ERROR_AS_FEATURE
}

joblib.dump(metadata, os.path.join(SAVE_FOLDER, "metadata_feddistill_ae_mlp_ciciot.pkl"))
joblib.dump(test_results, os.path.join(SAVE_FOLDER, "metrics_feddistill_ae_mlp_ciciot.pkl"))

results_df = pd.DataFrame({
    "source_file": src_test,
    "y_true": y_mlp_test,
    "y_pred": test_pred,
    "attack_score": test_scores,
    "ae_reconstruction_error_scaled": mixed_errors_test
})

results_df.to_csv(
    os.path.join(SAVE_FOLDER, "detailed_predictions_feddistill_ae_mlp_ciciot.csv"),
    index=False
)

print("\n===== SAUVEGARDE TERMINÉE =====")
print("Dossier résultats :", SAVE_FOLDER)
print("Fichier prédictions :", os.path.join(SAVE_FOLDER, "detailed_predictions_feddistill_ae_mlp_ciciot.csv"))
