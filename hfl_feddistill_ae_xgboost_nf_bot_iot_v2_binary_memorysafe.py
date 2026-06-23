import os
import copy
import random
import pickle
import sys
import gc

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, classification_report, roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =========================================================
# 1. PARAMETRES - NF-BoT-IoT-v2 HFL+FedDistill -> AE -> XGBoost
# =========================================================
DATA_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/NF-BoT-IoT"
PARQUET_FILE = os.path.join(DATA_FOLDER, "NF-BoT-IoT-V2.parquet")
FEATURE_DESCRIPTION_CSV = os.path.join(DATA_FOLDER, "NetFlow v2 Features.csv")

SAVE_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/Tests/NF-BoT-IoT/HFL_FedDistill_AE_XGBoost_BINARY_MEMORY_SAFE"
os.makedirs(SAVE_FOLDER, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30
VALID_SIZE = 0.20
AE_VALID_SIZE = 0.20

NUM_CLIENTS = 5
CLIENT_DATA_PROPORTIONS = [
    0.40,
    0.25, 
    0.18, 
    0.12, 
    0.05
]

NUM_EDGE_SERVERS = 2
CLIENT_EDGE_ASSIGNMENTS = [
    0, 
    0, 
    1, 
    1, 
    1
]

GLOBAL_ROUNDS_AE = 8
LOCAL_EPOCHS_AE = 5
AE_DISTILL_EPOCHS = 2
AE_DISTILL_ALPHA = 0.40

BATCH_SIZE = 1024
LEARNING_RATE_AE = 1e-3

PUBLIC_DISTILL_SIZE = 30_000
DISTILL_ALPHA_XGB = 0.65
XGB_LOCAL_ROUNDS = 450
XGB_STUDENT_ROUNDS = 650

# Dataset tres desequilibre: on limite et on force un echantillon contenant assez de Benign.
PARQUET_BATCH_SIZE = 250_000
ROWS_PER_BATCH_SAMPLE = 40_000
MAX_TOTAL_ROWS = 2_500_000
MIN_BENIGN_ROWS_TARGET = 80_000

AUTO_THRESHOLD_PERCENTILES = np.concatenate([
    np.arange(70.0, 99.0, 0.5),
    np.arange(99.0, 99.95, 0.05),
])

MAX_ALLOWED_FPR = 0.08
XGB_THRESHOLD_GRID = np.arange(0.01, 0.96, 0.005)

USE_RECONSTRUCTION_ERROR_AS_FEATURE = True
USE_AE_DECISION_AS_XGB_FEATURE = True
NORMAL_LABEL = 0
ATTACK_LABEL = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, **kwargs)


def save_object(obj, path):
    if joblib is not None:
        joblib.dump(obj, path)
        return
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def require_xgboost():
    if xgb is None:
        raise ImportError(
            "xgboost est requis. Installez-le avec :\n"
            f'  & "{sys.executable}" -m pip install xgboost'
        )


def validate_config():
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(f"Fichier introuvable : {PARQUET_FILE}")
    if len(CLIENT_DATA_PROPORTIONS) != NUM_CLIENTS:
        raise ValueError("CLIENT_DATA_PROPORTIONS doit avoir la meme longueur que NUM_CLIENTS.")
    if not np.isclose(sum(CLIENT_DATA_PROPORTIONS), 1.0):
        raise ValueError("La somme de CLIENT_DATA_PROPORTIONS doit etre egale a 1.0.")
    if len(CLIENT_EDGE_ASSIGNMENTS) != NUM_CLIENTS:
        raise ValueError("CLIENT_EDGE_ASSIGNMENTS doit avoir la meme longueur que NUM_CLIENTS.")
    if min(CLIENT_EDGE_ASSIGNMENTS) < 0 or max(CLIENT_EDGE_ASSIGNMENTS) >= NUM_EDGE_SERVERS:
        raise ValueError("Chaque edge id doit etre entre 0 et NUM_EDGE_SERVERS - 1.")


# =========================================================
# 2. CHARGEMENT PARQUET MEMORY SAFE
# =========================================================
LABEL_CANDIDATES = ["Label", "label", "TARGET", "target", "class", "Class"]
ATTACK_TEXT_CANDIDATES = ["Attack", "attack", "Attack_cat", "attack_cat", "Category", "category"]
NORMAL_STRINGS = {"0", "normal", "benign", "legitimate", "false", "no", "none"}

DROP_COLS_ALWAYS = [
    "id", "ID", "Flow ID", "flow_id", "FlowID", "Timestamp", "timestamp", "ts",
    "Src IP", "Dst IP", "Source IP", "Destination IP", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
    "src_ip", "dst_ip", "src_mac", "dst_mac", "Unnamed: 0",
]


def find_column(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def make_binary_labels(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return (numeric.fillna(0).astype(float) != 0).astype(np.int64).to_numpy()
    values = series.astype(str).str.strip().str.lower()
    return (~values.isin(NORMAL_STRINGS)).astype(np.int64).to_numpy()


def clean_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()
    label_col = find_column(chunk.columns, LABEL_CANDIDATES)
    if label_col is None:
        attack_col = find_column(chunk.columns, ATTACK_TEXT_CANDIDATES)
        if attack_col is None:
            raise ValueError("Aucune colonne Label/Attack trouvee dans le parquet.")
        label_col = attack_col

    y = make_binary_labels(chunk[label_col])
    attack_col = find_column(chunk.columns, ATTACK_TEXT_CANDIDATES)
    original_labels = chunk[attack_col].astype(str).to_numpy() if attack_col else chunk[label_col].astype(str).to_numpy()

    drop_cols = [label_col]
    drop_cols += [col for col in DROP_COLS_ALWAYS if col in chunk.columns]
    if attack_col is not None and attack_col != label_col:
        drop_cols.append(attack_col)
    drop_cols = sorted(set(drop_cols))

    feature_cols = [col for col in chunk.columns if col not in drop_cols]
    numeric_cols = [col for col in feature_cols if is_numeric_dtype(chunk[col])]
    chunk = chunk[numeric_cols].copy()
    chunk["__y__"] = y
    chunk["__original_label__"] = original_labels
    return chunk, label_col, attack_col, numeric_cols


def balanced_sample_chunk(chunk, batch_idx):
    if len(chunk) <= ROWS_PER_BATCH_SAMPLE:
        return chunk

    rng = np.random.default_rng(RANDOM_STATE + batch_idx)
    benign_idx = np.where(chunk["__y__"].to_numpy() == NORMAL_LABEL)[0]
    attack_idx = np.where(chunk["__y__"].to_numpy() == ATTACK_LABEL)[0]

    benign_take = min(len(benign_idx), max(1, ROWS_PER_BATCH_SAMPLE // 2))
    attack_take = min(len(attack_idx), ROWS_PER_BATCH_SAMPLE - benign_take)

    selected = []
    if benign_take > 0:
        selected.extend(rng.choice(benign_idx, size=benign_take, replace=False).tolist())
    if attack_take > 0:
        selected.extend(rng.choice(attack_idx, size=attack_take, replace=False).tolist())

    if len(selected) < ROWS_PER_BATCH_SAMPLE:
        remaining = np.setdiff1d(np.arange(len(chunk)), np.array(selected), assume_unique=False)
        take = min(ROWS_PER_BATCH_SAMPLE - len(selected), len(remaining))
        if take > 0:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())

    rng.shuffle(selected)
    return chunk.iloc[selected].copy()


def read_parquet_sample_pyarrow(path):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    chunks = []
    total_rows = 0
    total_benign = 0
    label_col_seen = None
    attack_col_seen = None
    common_numeric_cols = None

    batches = parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE)
    for batch_idx, batch in enumerate(progress(batches, desc="Lecture parquet batches", leave=True)):
        chunk = batch.to_pandas()
        chunk, label_col, attack_col, numeric_cols = clean_chunk(chunk)
        chunk = balanced_sample_chunk(chunk, batch_idx)

        remaining = MAX_TOTAL_ROWS - total_rows if MAX_TOTAL_ROWS is not None else len(chunk)
        if remaining <= 0 and total_benign >= MIN_BENIGN_ROWS_TARGET:
            break
        if remaining > 0 and len(chunk) > remaining:
            chunk = chunk.sample(n=remaining, random_state=RANDOM_STATE + 1000 + batch_idx)

        label_col_seen = label_col_seen or label_col
        attack_col_seen = attack_col_seen or attack_col
        common_numeric_cols = set(numeric_cols) if common_numeric_cols is None else common_numeric_cols.intersection(numeric_cols)

        chunks.append(chunk)
        total_rows += len(chunk)
        total_benign += int(np.sum(chunk["__y__"].to_numpy() == NORMAL_LABEL))
        del batch, chunk
        gc.collect()

        if MAX_TOTAL_ROWS is not None and total_rows >= MAX_TOTAL_ROWS and total_benign >= MIN_BENIGN_ROWS_TARGET:
            print("MAX_TOTAL_ROWS atteint avec assez de Benign.")
            break

    if not chunks:
        raise ValueError("Aucune donnee chargee depuis le parquet.")

    common_numeric_cols = sorted(common_numeric_cols)
    aligned_chunks = []
    for chunk in progress(chunks, desc="Alignement colonnes", leave=True):
        aligned_chunks.append(chunk[common_numeric_cols + ["__y__", "__original_label__"]])

    df = pd.concat(aligned_chunks, ignore_index=True)
    return df, label_col_seen, attack_col_seen, common_numeric_cols


def read_parquet_sample_pandas(path):
    df = pd.read_parquet(path)
    df, label_col, attack_col, numeric_cols = clean_chunk(df)
    df = balanced_sample_chunk(df, 0)
    if MAX_TOTAL_ROWS is not None and len(df) > MAX_TOTAL_ROWS:
        df = df.sample(n=MAX_TOTAL_ROWS, random_state=RANDOM_STATE)
    return df, label_col, attack_col, numeric_cols


def load_dataset():
    print("Device utilise :", DEVICE)
    print("\nLecture dataset :", PARQUET_FILE)
    if os.path.exists(FEATURE_DESCRIPTION_CSV):
        print("Info : NetFlow v2 Features.csv est un dictionnaire de colonnes.")

    try:
        df, label_col, attack_col, numeric_cols = read_parquet_sample_pyarrow(PARQUET_FILE)
        engine = "pyarrow batches"
    except Exception as exc:
        print("Lecture par batches indisponible :", exc)
        print("Fallback pandas read_parquet. Si RAM insuffisante, installez pyarrow.")
        df, label_col, attack_col, numeric_cols = read_parquet_sample_pandas(PARQUET_FILE)
        engine = "pandas"

    y = df.pop("__y__").astype(np.int64).to_numpy()
    original_labels = df.pop("__original_label__").astype(str).to_numpy()
    numeric_cols = list(df.columns)

    print("\n===== DATASET CHARGE =====")
    print("Moteur lecture :", engine)
    print("Lignes gardees :", len(df))
    print("Colonnes features numeriques :", len(numeric_cols))
    print("Colonne label :", label_col)
    print("Colonne attaque descriptive :", attack_col)
    print(pd.Series(y).value_counts().sort_index())
    return df, y, original_labels, label_col, attack_col, numeric_cols


# =========================================================
# 3. PRETRAITEMENT
# =========================================================
def fit_numeric_preprocessor(df, feature_cols, idx):
    arrays, medians, kept_cols = [], {}, []
    for col in progress(feature_cols, desc="Fit numeric float32", leave=True):
        s = pd.to_numeric(df[col].iloc[idx], errors="coerce")
        arr = s.to_numpy(dtype=np.float32, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        if np.all(np.isnan(arr)):
            continue
        median = float(np.nanmedian(arr))
        if not np.isfinite(median):
            median = 0.0
        arr = np.nan_to_num(arr, nan=median, posinf=median, neginf=median).astype(np.float32, copy=False)
        if float(np.min(arr)) == float(np.max(arr)):
            continue
        medians[col] = median
        kept_cols.append(col)
        arrays.append(arr)

    if not arrays:
        raise ValueError("Aucune colonne numerique exploitable.")

    X_raw = np.column_stack(arrays).astype(np.float32, copy=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, {"numeric_cols": kept_cols, "medians": medians, "scaler": scaler}


def transform_numeric(df, idx, preprocessor):
    arrays = []
    for col in progress(preprocessor["numeric_cols"], desc="Transform numeric float32", leave=True):
        if col in df.columns:
            s = pd.to_numeric(df[col].iloc[idx], errors="coerce")
            arr = s.to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.full(len(idx), np.nan, dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan
        median = float(preprocessor["medians"].get(col, 0.0))
        arr = np.nan_to_num(arr, nan=median, posinf=median, neginf=median).astype(np.float32, copy=False)
        arrays.append(arr)

    X_raw = np.column_stack(arrays).astype(np.float32, copy=False)
    X = preprocessor["scaler"].transform(X_raw).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def print_distribution(title, y):
    print(f"\n===== {title} =====")
    print(pd.Series(y).value_counts().sort_index())
    print("Benign :", int(np.sum(y == NORMAL_LABEL)))
    print("Attack :", int(np.sum(y == ATTACK_LABEL)))


# =========================================================
# 4. HFL + AUTOENCODER
# =========================================================
def split_clients_unsupervised(X, proportions):
    idx = np.random.permutation(len(X))
    X = X[idx]
    counts = [int(len(X) * p) for p in proportions]
    counts[-1] = len(X) - sum(counts[:-1])
    clients, start = [], 0
    for count in counts:
        clients.append(X[start:start + count])
        start += count
    return clients


def split_clients_supervised(X, y, proportions):
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    counts = [int(len(X) * p) for p in proportions]
    counts[-1] = len(X) - sum(counts[:-1])
    clients, start = [], 0
    for count in counts:
        clients.append((X[start:start + count], y[start:start + count]))
        start += count
    return clients


def weighted_average_state_dicts(state_dicts, sizes):
    total_size = sum(sizes)
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        if not torch.is_floating_point(avg_state[key]):
            avg_state[key] = state_dicts[0][key]
            continue
        avg_state[key] = torch.zeros_like(avg_state[key])
        for state, size in zip(state_dicts, sizes):
            avg_state[key] += state[key] * (size / total_size)
    return avg_state


def hfl_aggregate(local_weights, local_sizes):
    edge_weights, edge_sizes = [], []
    for edge_id in range(NUM_EDGE_SERVERS):
        client_ids = [i for i, edge in enumerate(CLIENT_EDGE_ASSIGNMENTS) if edge == edge_id and local_sizes[i] > 0]
        if not client_ids:
            continue

        edge_weights.append(
            weighted_average_state_dicts(
                [local_weights[i] for i in client_ids], 
                [local_sizes[i] for i in client_ids]
            )
        )

        edge_sizes.append(sum(local_sizes[i] for i in client_ids))

    return weighted_average_state_dicts(
        edge_weights, 
        edge_sizes
    )


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def make_ae_loader(X, shuffle=True):
    t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    return DataLoader(TensorDataset(t, t), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_local_autoencoder(global_model, client_X, client_idx):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()
    loader = make_ae_loader(client_X, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE_AE)
    losses = []

    for epoch_idx in range(LOCAL_EPOCHS_AE):
        running_loss = 0.0
        iterator = progress(loader, desc=f"AE client {client_idx + 1} E{epoch_idx + 1}/{LOCAL_EPOCHS_AE}", leave=False)
        for batch_x, batch_target in iterator:
            batch_x = batch_x.to(DEVICE)
            batch_target = batch_target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(local_model(batch_x), batch_target)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
        losses.append(running_loss / len(loader.dataset))

    return local_model.state_dict(), len(client_X), float(np.mean(losses))


def hfl_collect_ae_reconstruction_targets(client_models, X_public, client_sizes):
    loader = DataLoader(torch.from_numpy(np.asarray(X_public, dtype=np.float32)), batch_size=BATCH_SIZE, shuffle=False)
    client_reconstructions = []
    for model in progress(client_models, desc="Collecte reconstructions AE", leave=False):
        model.eval()
        batches = []
        with torch.no_grad():
            for batch_x in loader:
                batches.append(model(batch_x.to(DEVICE)).cpu().numpy())
        client_reconstructions.append(np.vstack(batches))

    edge_reconstructions, edge_sizes = [], []
    for edge_id in range(NUM_EDGE_SERVERS):
        client_ids = [i for i, edge in enumerate(CLIENT_EDGE_ASSIGNMENTS) if edge == edge_id and client_sizes[i] > 0]
        if not client_ids:
            continue
        edge_size = sum(client_sizes[i] for i in client_ids)
        edge_target = np.zeros_like(client_reconstructions[0], dtype=np.float32)
        for client_id in client_ids:
            edge_target += client_reconstructions[client_id] * (client_sizes[client_id] / edge_size)
        edge_reconstructions.append(edge_target)
        edge_sizes.append(edge_size)

    total_size = sum(edge_sizes)
    global_target = np.zeros_like(edge_reconstructions[0], dtype=np.float32)
    for target, size in zip(edge_reconstructions, edge_sizes):
        global_target += target * (size / total_size)
    return global_target.astype(np.float32)


def distill_global_ae(global_model, X_public, reconstruction_targets):
    global_model = copy.deepcopy(global_model).to(DEVICE)
    global_model.train()
    dataset = TensorDataset(torch.from_numpy(np.asarray(X_public, dtype=np.float32)), torch.from_numpy(np.asarray(reconstruction_targets, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE_AE)
    losses = []

    for epoch_idx in range(AE_DISTILL_EPOCHS):
        running_loss = 0.0
        iterator = progress(loader, desc=f"Distillation AE E{epoch_idx + 1}/{AE_DISTILL_EPOCHS}", leave=False)
        for batch_x, batch_soft in iterator:
            batch_x = batch_x.to(DEVICE)
            batch_soft = batch_soft.to(DEVICE)
            optimizer.zero_grad()
            output = global_model(batch_x)
            loss = (1.0 - AE_DISTILL_ALPHA) * criterion(output, batch_x) + AE_DISTILL_ALPHA * criterion(output, batch_soft)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
        losses.append(running_loss / len(loader.dataset))
    return global_model.state_dict(), float(np.mean(losses))


def compute_reconstruction_errors(model, X):
    model.eval()
    loader = DataLoader(torch.from_numpy(np.asarray(X, dtype=np.float32)), batch_size=BATCH_SIZE, shuffle=False)
    errors = []
    with torch.no_grad():
        for batch_x in progress(loader, desc="Erreurs reconstruction", leave=False):
            batch_x = batch_x.to(DEVICE)
            errors.extend(torch.mean((batch_x - model(batch_x)) ** 2, dim=1).cpu().numpy())
    return np.array(errors, dtype=np.float32)


def choose_auto_threshold_percentile(normal_reference_errors, validation_errors, validation_y):
    best = {"percentile": None, "threshold": None, "f1": -1.0, "recall": 0.0, "fpr": 1.0}
    for percentile in AUTO_THRESHOLD_PERCENTILES:
        threshold = float(np.percentile(normal_reference_errors, percentile))
        pred = (validation_errors > threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(validation_y, pred, labels=[NORMAL_LABEL, ATTACK_LABEL]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rec = recall_score(validation_y, pred, zero_division=0)
        f1 = f1_score(validation_y, pred, zero_division=0)
        if f1 > best["f1"] or (np.isclose(f1, best["f1"]) and rec > best["recall"]):
            best = {"percentile": float(percentile), "threshold": threshold, "f1": float(f1), "recall": float(rec), "fpr": float(fpr)}
    return best


def extract_ae_features_and_detection(model, X, threshold, normal_error_std):
    model.eval()
    loader = DataLoader(torch.from_numpy(np.asarray(X, dtype=np.float32)), batch_size=BATCH_SIZE, shuffle=False)
    all_z, all_errors = [], []
    with torch.no_grad():
        for batch_x in progress(loader, desc="Features AE", leave=False):
            batch_x = batch_x.to(DEVICE)
            z = model.encode(batch_x)
            x_hat = model(batch_x)
            err = torch.mean((batch_x - x_hat) ** 2, dim=1, keepdim=True)
            all_z.append(z.cpu().numpy())
            all_errors.append(err.cpu().numpy())

    z_all = np.vstack(all_z).astype(np.float32)
    err_all = np.vstack(all_errors).astype(np.float32)
    features = [z_all]
    if USE_RECONSTRUCTION_ERROR_AS_FEATURE:
        features.append(err_all)
    ae_pred = (err_all > threshold).astype(np.float32)
    ae_score = ((err_all - threshold) / max(normal_error_std, 1e-12)).astype(np.float32)
    if USE_AE_DECISION_AS_XGB_FEATURE:
        features.extend([ae_pred, ae_score])
    return np.hstack(features).astype(np.float32), err_all.ravel(), ae_pred.ravel().astype(np.int64), ae_score.ravel()


# =========================================================
# 5. XGBOOST + FEDDISTILL
# =========================================================
def compute_binary_sample_weights(y):
    counts = np.bincount(y.astype(np.int64), minlength=2)
    total = counts.sum()
    weights = total / (2.0 * np.maximum(counts, 1))
    return weights[y.astype(np.int64)].astype(np.float32)


def make_xgb_params(y):
    counts = np.bincount(y.astype(np.int64), minlength=2)
    scale_pos_weight = float(counts[NORMAL_LABEL] / max(counts[ATTACK_LABEL], 1))
    return {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "auc", "logloss"],
        "eta": 0.035,
        "max_depth": 8,
        "min_child_weight": 2.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "gamma": 0.05,
        "lambda": 1.5,
        "alpha": 0.05,
        "tree_method": "hist",
        "max_bin": 256,
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
        "verbosity": 0,
    }


def train_local_xgb(client_X, client_y, client_idx):
    dtrain = xgb.DMatrix(client_X, label=client_y, weight=compute_binary_sample_weights(client_y))
    params = make_xgb_params(client_y)
    evals_result = {}
    model = xgb.train(params, dtrain, num_boost_round=XGB_LOCAL_ROUNDS, evals=[(dtrain, f"client_{client_idx + 1}")], evals_result=evals_result, verbose_eval=False)
    return model, len(client_y), float(evals_result[f"client_{client_idx + 1}"]["aucpr"][-1])


def sample_public_balanced(X, y, max_size, random_state=42):
    rng = np.random.default_rng(random_state)
    selected = []
    classes = np.unique(y)
    per_class = max(1, max_size // len(classes))
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        selected.extend(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False).tolist())
    if len(selected) < min(max_size, len(y)):
        remaining = np.setdiff1d(np.arange(len(y)), np.array(selected), assume_unique=False)
        take = min(max_size - len(selected), len(remaining))
        if take > 0:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    selected = np.array(selected)
    rng.shuffle(selected)
    return X[selected], y[selected]


def hfl_collect_xgb_soft_targets(client_models, X_public, client_sizes):
    dpublic = xgb.DMatrix(X_public)
    client_probs = [model.predict(dpublic).astype(np.float32) for model in progress(client_models, desc="Soft labels XGBoost clients", leave=False)]
    edge_probs, edge_sizes = [], []
    for edge_id in range(NUM_EDGE_SERVERS):
        client_ids = [i for i, edge in enumerate(CLIENT_EDGE_ASSIGNMENTS) if edge == edge_id and client_sizes[i] > 0]
        if not client_ids:
            continue
        edge_size = sum(client_sizes[i] for i in client_ids)
        edge_soft = np.zeros_like(client_probs[0], dtype=np.float32)
        for client_id in client_ids:
            edge_soft += client_probs[client_id] * (client_sizes[client_id] / edge_size)
        edge_probs.append(edge_soft)
        edge_sizes.append(edge_size)
    total_size = sum(edge_sizes)
    soft_targets = np.zeros_like(edge_probs[0], dtype=np.float32)
    for probs, size in zip(edge_probs, edge_sizes):
        soft_targets += probs * (size / total_size)
    return np.clip(soft_targets, 1e-5, 1.0 - 1e-5).astype(np.float32)


def train_distilled_xgb(X_public, y_public, soft_targets):
    soft_labels = ((1.0 - DISTILL_ALPHA_XGB) * y_public.astype(np.float32)) + (DISTILL_ALPHA_XGB * soft_targets)
    dtrain = xgb.DMatrix(X_public, label=soft_labels, weight=compute_binary_sample_weights(y_public))
    model = xgb.train(make_xgb_params(y_public), dtrain, num_boost_round=XGB_STUDENT_ROUNDS, evals=[(dtrain, "student")], verbose_eval=False)
    return model


def predict_xgb(model, X):
    return model.predict(xgb.DMatrix(X)).astype(np.float32)


def choose_xgb_threshold(y_true, scores):
    best, fallback = None, None
    for threshold in XGB_THRESHOLD_GRID:
        pred = (scores >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[NORMAL_LABEL, ATTACK_LABEL]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        info = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
            "fpr": float(fpr),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        score = 0.45 * info["f1"] + 0.35 * info["recall"] + 0.20 * info["accuracy"]
        fallback_score = 0.55 * info["f1"] + 0.30 * info["recall"] + 0.15 * info["accuracy"]
        info["selection_score"] = float(score)
        if fallback is None or fallback_score > fallback["fallback_score"]:
            fallback = dict(info)
            fallback["fallback_score"] = float(fallback_score)
        if fpr <= MAX_ALLOWED_FPR and (best is None or score > best["selection_score"]):
            best = info
    if best is None:
        fallback["note"] = "Aucun seuil ne respecte MAX_ALLOWED_FPR; seuil choisi par compromis F1/Recall/Accuracy."
        return fallback
    best["note"] = f"Seuil choisi par compromis F1/Recall/Accuracy avec FPR <= {MAX_ALLOWED_FPR}."
    return best


def evaluate_binary(name, y_true, y_pred, scores=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NORMAL_LABEL, ATTACK_LABEL]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, scores) if scores is not None else np.nan
    except Exception:
        auc = np.nan

    print(f"\n===== RESULTATS {name} =====")
    print("Accuracy              :", acc)
    print("Precision             :", prec)
    print("Recall / DetectionRate:", rec)
    print("F1-score              :", f1)
    print("F2-score              :", f2)
    print("Specificity           :", specificity)
    print("False Positive Rate   :", fpr)
    print("False Negative Rate   :", fnr)
    print("ROC-AUC               :", auc)
    print("Matrice de confusion  :")
    print(np.array([[tn, fp], [fn, tp]]))
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=4, zero_division=0))
    return {
        "accuracy": float(acc), "precision": float(prec), "recall_detection_rate": float(rec),
        "f1_score": float(f1), "f2_score": float(f2), "specificity": float(specificity),
        "fpr": float(fpr), "fnr": float(fnr), "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# =========================================================
# 6. PIPELINE
# =========================================================
def main():
    validate_config()
    require_xgboost()

    df_all, y_all, original_labels, label_col, attack_col, feature_cols = load_dataset()
    print_distribution("DISTRIBUTION COMPLETE", y_all)

    all_idx = np.arange(len(y_all))
    train_full_idx, test_idx = train_test_split(all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=VALID_SIZE, random_state=RANDOM_STATE, stratify=y_all[train_full_idx])

    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    labels_test = original_labels[test_idx]
    print_distribution("TRAIN", y_train)
    print_distribution("VALIDATION", y_val)
    print_distribution("TEST", y_test)

    print("\n===== PRETRAITEMENT MEMORY SAFE =====")
    X_train, preprocessor = fit_numeric_preprocessor(df_all, feature_cols, train_idx)
    X_val = transform_numeric(df_all, val_idx, preprocessor)
    X_test = transform_numeric(df_all, test_idx, preprocessor)
    del df_all
    gc.collect()

    X_train_normal = X_train[y_train == NORMAL_LABEL]
    if len(X_train_normal) == 0:
        raise ValueError("Aucune donnee Benign pour entrainer l'autoencodeur.")
    X_ae_train, X_ae_val = train_test_split(X_train_normal, test_size=AE_VALID_SIZE, random_state=RANDOM_STATE)
    clients_ae = split_clients_unsupervised(X_ae_train, CLIENT_DATA_PROPORTIONS)

    input_dim = X_train.shape[1]
    global_ae = Autoencoder(input_dim).to(DEVICE)
    best_ae_val_loss = float("inf")
    best_ae_path = os.path.join(SAVE_FOLDER, "best_hfl_feddistill_ae_nf_bot_iot_v2_binary.pth")

    print("\n===== ENTRAINEMENT HFL + FEDDISTILL + AE =====")
    for round_idx in progress(range(GLOBAL_ROUNDS_AE), desc="Rounds AE", leave=True):
        print(f"\n--- Round AE {round_idx + 1}/{GLOBAL_ROUNDS_AE} ---")
        local_models, local_weights, local_sizes, local_losses = [], [], [], []
        for client_idx, client_X in enumerate(clients_ae):
            weights, size, loss = train_local_autoencoder(global_ae, client_X, client_idx)
            client_model = Autoencoder(input_dim).to(DEVICE)
            client_model.load_state_dict(weights)
            client_model.eval()
            local_models.append(client_model)
            local_weights.append(weights)
            local_sizes.append(size)
            local_losses.append(loss)

        global_ae.load_state_dict(hfl_aggregate(local_weights, local_sizes))
        ae_targets = hfl_collect_ae_reconstruction_targets(local_models, X_ae_val, local_sizes)
        distilled_weights, ae_distill_loss = distill_global_ae(global_ae, X_ae_val, ae_targets)
        global_ae.load_state_dict(distilled_weights)
        val_loss = float(np.mean(compute_reconstruction_errors(global_ae, X_ae_val)))
        print(f"Loss moyenne clients AE : {np.mean(local_losses):.6f}")
        print(f"Loss distillation AE    : {ae_distill_loss:.6f}")
        print(f"Validation loss AE      : {val_loss:.6f}")
        if val_loss < best_ae_val_loss:
            best_ae_val_loss = val_loss
            torch.save(global_ae.state_dict(), best_ae_path)
            print("Meilleur AE sauvegarde.")

    best_ae = Autoencoder(input_dim).to(DEVICE)
    best_ae.load_state_dict(torch.load(best_ae_path, map_location=DEVICE))
    best_ae.eval()

    print("\n===== SEUIL AE =====")
    normal_reference_errors = compute_reconstruction_errors(best_ae, X_ae_train)
    val_errors_for_threshold = compute_reconstruction_errors(best_ae, X_val)
    ae_threshold_info = choose_auto_threshold_percentile(normal_reference_errors, val_errors_for_threshold, y_val)
    AE_THRESHOLD = ae_threshold_info["threshold"]
    AE_NORMAL_ERROR_STD = float(np.std(normal_reference_errors))
    print("Percentile AE :", ae_threshold_info["percentile"])
    print("Seuil AE      :", AE_THRESHOLD)
    print("AE val F1     :", ae_threshold_info["f1"])
    print("AE val Recall :", ae_threshold_info["recall"])
    print("AE val FPR    :", ae_threshold_info["fpr"])

    print("\n===== EXTRACTION FEATURES AE POUR XGBOOST =====")
    X_xgb_train, train_errors, train_ae_pred, train_ae_score = extract_ae_features_and_detection(best_ae, X_train, AE_THRESHOLD, AE_NORMAL_ERROR_STD)
    X_xgb_val, val_errors, val_ae_pred, val_ae_score = extract_ae_features_and_detection(best_ae, X_val, AE_THRESHOLD, AE_NORMAL_ERROR_STD)
    X_xgb_test, test_errors, test_ae_pred, test_ae_score = extract_ae_features_and_detection(best_ae, X_test, AE_THRESHOLD, AE_NORMAL_ERROR_STD)
    del X_train, X_val, X_test, X_ae_train, X_ae_val
    gc.collect()

    xgb_feature_scaler = StandardScaler()
    X_xgb_train = xgb_feature_scaler.fit_transform(X_xgb_train).astype(np.float32)
    X_xgb_val = xgb_feature_scaler.transform(X_xgb_val).astype(np.float32)
    X_xgb_test = xgb_feature_scaler.transform(X_xgb_test).astype(np.float32)

    clients_xgb = split_clients_supervised(X_xgb_train, y_train, CLIENT_DATA_PROPORTIONS)
    X_public, y_public = sample_public_balanced(X_xgb_val, y_val, PUBLIC_DISTILL_SIZE, RANDOM_STATE)

    print("\n===== ENTRAINEMENT HFL + FEDDISTILL + XGBOOST =====")
    local_models, local_sizes, local_aucpr = [], [], []
    for client_idx, (client_X, client_y) in enumerate(progress(clients_xgb, desc="Clients XGBoost", leave=True)):
        model, size, aucpr = train_local_xgb(client_X, client_y, client_idx)
        local_models.append(model)
        local_sizes.append(size)
        local_aucpr.append(aucpr)
        print(f"Client {client_idx + 1} -> Edge {CLIENT_EDGE_ASSIGNMENTS[client_idx] + 1} | size={size} | aucpr={aucpr:.6f}")

    soft_targets = hfl_collect_xgb_soft_targets(local_models, X_public, local_sizes)
    global_xgb = train_distilled_xgb(X_public, y_public, soft_targets)
    global_xgb_path = os.path.join(SAVE_FOLDER, "hfl_feddistill_xgboost_nf_bot_iot_v2_binary.json")
    global_xgb.save_model(global_xgb_path)

    print("\n===== SEUIL FINAL XGBOOST =====")
    val_scores = predict_xgb(global_xgb, X_xgb_val)
    threshold_info = choose_xgb_threshold(y_val, val_scores)
    XGB_DECISION_THRESHOLD = threshold_info["threshold"]
    print("Seuil XGBoost choisi :", XGB_DECISION_THRESHOLD)
    print("Validation Accuracy  :", threshold_info["accuracy"])
    print("Validation Recall    :", threshold_info["recall"])
    print("Validation F1        :", threshold_info["f1"])
    print("Validation FPR       :", threshold_info["fpr"])
    print("Note seuil           :", threshold_info["note"])

    print("\n===== TEST FINAL HFL+FedDistill -> AE -> XGBoost BINAIRE =====")
    test_scores = predict_xgb(global_xgb, X_xgb_test)
    test_pred_threshold = (test_scores >= XGB_DECISION_THRESHOLD).astype(np.int64)
    ae_metrics = evaluate_binary("AE SEUL - SEUIL AUTO", y_test, test_ae_pred, scores=test_errors)
    final_metrics = evaluate_binary("HFL+FedDistill -> AE -> XGBoost BINAIRE", y_test, test_pred_threshold, scores=test_scores)

    torch.save(best_ae.state_dict(), os.path.join(SAVE_FOLDER, "hfl_feddistill_ae_nf_bot_iot_v2_binary_final.pth"))
    save_object(preprocessor, os.path.join(SAVE_FOLDER, "preprocessor_nf_bot_iot_v2_binary.pkl"))
    save_object(xgb_feature_scaler, os.path.join(SAVE_FOLDER, "xgb_feature_scaler_nf_bot_iot_v2_binary.pkl"))
    save_object(ae_threshold_info, os.path.join(SAVE_FOLDER, "ae_threshold_info_nf_bot_iot_v2_binary.pkl"))
    save_object(threshold_info, os.path.join(SAVE_FOLDER, "xgb_threshold_info_nf_bot_iot_v2_binary.pkl"))

    metadata = {
        "architecture": "NF-BoT-IoT-v2 HFL + FedDistill -> AE -> XGBoost binary memory-safe",
        "data_folder": DATA_FOLDER,
        "parquet_file": PARQUET_FILE,
        "label_col": label_col,
        "attack_col_excluded_from_features": attack_col,
        "num_clients": NUM_CLIENTS,
        "num_edge_servers": NUM_EDGE_SERVERS,
        "client_edge_assignments": CLIENT_EDGE_ASSIGNMENTS,
        "client_data_proportions": CLIENT_DATA_PROPORTIONS,
        "max_total_rows": MAX_TOTAL_ROWS,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
        "rows_per_batch_sample": ROWS_PER_BATCH_SAMPLE,
        "min_benign_rows_target": MIN_BENIGN_ROWS_TARGET,
        "global_rounds_ae": GLOBAL_ROUNDS_AE,
        "local_epochs_ae": LOCAL_EPOCHS_AE,
        "xgb_local_rounds": XGB_LOCAL_ROUNDS,
        "xgb_student_rounds": XGB_STUDENT_ROUNDS,
        "distill_alpha_xgb": DISTILL_ALPHA_XGB,
        "max_allowed_fpr": MAX_ALLOWED_FPR,
        "best_ae_val_loss": best_ae_val_loss,
        "ae_threshold": AE_THRESHOLD,
        "xgb_decision_threshold": XGB_DECISION_THRESHOLD,
        "input_dim": input_dim,
        "xgb_input_dim": X_xgb_train.shape[1],
        "numeric_cols": preprocessor["numeric_cols"],
    }
    all_metrics = {
        "ae_test_metrics": ae_metrics,
        "final_xgb_metrics": final_metrics,
        "ae_threshold_info": ae_threshold_info,
        "xgb_threshold_info": threshold_info,
    }
    save_object(metadata, os.path.join(SAVE_FOLDER, "metadata_nf_bot_iot_v2_binary.pkl"))
    save_object(all_metrics, os.path.join(SAVE_FOLDER, "metrics_nf_bot_iot_v2_binary.pkl"))

    results_df = pd.DataFrame({
        "label_original": labels_test,
        "y_true": y_test,
        "ae_pred": test_ae_pred,
        "xgb_pred_threshold": test_pred_threshold,
        "reconstruction_error": test_errors,
        "ae_score": test_ae_score,
        "xgb_attack_score": test_scores,
    })
    results_path = os.path.join(SAVE_FOLDER, "detailed_predictions_nf_bot_iot_v2_binary.csv")
    results_df.to_csv(results_path, index=False)

    print("\n===== SAUVEGARDE TERMINEE =====")
    print("Dossier resultats :", SAVE_FOLDER)
    print("Modele XGBoost    :", global_xgb_path)
    print("Predictions       :", results_path)
    print("Seuil AE          :", AE_THRESHOLD)
    print("Seuil XGBoost     :", XGB_DECISION_THRESHOLD)


if __name__ == "__main__":
    main()
