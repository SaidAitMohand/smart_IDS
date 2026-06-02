import os
import glob
import copy
import random
import joblib
import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =========================================================
# 1. PARAMETRES - CICIoT HFL + AE + MLP OPTIMISE
# =========================================================
BENIGN_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/CICIoT/Benign_Final"
MIXED_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/CICIoT/Test_CICIoT"

SAVE_FOLDER = "C:/Users/anis/Desktop/M2_RMSE/Memoire/DataSets/CICIoT/HFL_AE_MLP_OPTIMIZED_RECALL_FPR"
os.makedirs(SAVE_FOLDER, exist_ok=True)

RANDOM_STATE = 42
VALID_SIZE_AE = 0.20
MLP_TEST_SIZE = 0.30
MLP_VALID_SIZE = 0.20

NUM_CLIENTS = 5
NUM_EDGES = 2
CLIENT_DATA_PROPORTIONS = [0.40, 0.25, 0.18, 0.12, 0.05]
EDGE_GROUPS = [
    [0, 1],
    [2, 3, 4],
]

GLOBAL_ROUNDS = 10
LOCAL_EPOCHS_AE = 8
LOCAL_EPOCHS_MLP = 8

BATCH_SIZE = 512
LEARNING_RATE_AE = 1e-3
LEARNING_RATE_MLP = 8e-4
WEIGHT_DECAY_MLP = 1e-4

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
ATTACK_CLASS_WEIGHT_MULTIPLIER = 1.7

AUTO_THRESHOLD_PERCENTILES = np.concatenate([
    np.arange(70.0, 99.0, 0.5),
    np.arange(99.0, 99.95, 0.05),
])

MAX_ALLOWED_FPR = 0.08
MLP_THRESHOLD_GRID = np.arange(0.01, 0.96, 0.01)

USE_RECONSTRUCTION_ERROR_AS_FEATURE = True
USE_AE_DECISION_AS_MLP_FEATURE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilise :", DEVICE)

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

#Verification des clients et de la repartition des donnees par clients 
def validate_config():
    if len(CLIENT_DATA_PROPORTIONS) != NUM_CLIENTS:
        raise ValueError("CLIENT_DATA_PROPORTIONS doit avoir la meme taille que NUM_CLIENTS.")
    if any(p <= 0 for p in CLIENT_DATA_PROPORTIONS):
        raise ValueError("Toutes les proportions clients doivent etre positives.")
    if not np.isclose(sum(CLIENT_DATA_PROPORTIONS), 1.0):
        raise ValueError("La somme de CLIENT_DATA_PROPORTIONS doit etre egale a 1.0.")
    flat = [client for group in EDGE_GROUPS for client in group]
    if sorted(flat) != list(range(NUM_CLIENTS)):
        raise ValueError("EDGE_GROUPS doit couvrir exactement tous les clients 0..NUM_CLIENTS-1.")


validate_config()


# =========================================================
# 2. CHARGEMENT CSV
# =========================================================
def load_folder_with_labels(folder_path, infer_labels=True):
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"Aucun CSV trouve dans : {folder_path}")

    dfs = [] #Data Frames
    y = [] #Les Labels
    source_files = [] #Noms des fichiers sources

    print(f"\nChargement dossier : {folder_path}")
    print("Nombre de fichiers :", len(files))

    for file in files:
        filename = os.path.basename(file) # Nom du fichier
        df = pd.read_csv(file, low_memory=False) #Lecture des csv
        df.columns = df.columns.str.strip() #supprime les espaces inutiles
        dfs.append(df)
        source_files.extend([filename] * len(df))

        if infer_labels:
            label = 0 if "benign" in filename.lower() else 1 #donne des Labels
            label_name = "BENIGN" if label == 0 else "ATTACK"
            y.extend([label] * len(df))
            print(f"[OK] {filename} -> {df.shape} | {label_name}")
        else:
            print(f"[OK] {filename} -> {df.shape}")

    df_all = pd.concat(dfs, ignore_index=True) #fusionner les csv
    if infer_labels:
        return df_all, np.array(y, dtype=np.int64), source_files
    return df_all, source_files


# =========================================================
# 3. PRETRAITEMENT MEMOIRE-PRUDENT
# =========================================================
LABEL_CANDIDATES = ["label", "Label", "class", "Class", "Attack", "attack"]
DROP_COLS = [
    "Flow ID", "flow_id",
    "Timestamp", "timestamp",
    "Src IP", "Dst IP",
    "Source IP", "Destination IP",
    "Unnamed: 0",
]


def clean_column_list(df):
    drop = [c for c in LABEL_CANDIDATES if c in df.columns] #supprimes les Labels (pour eviter de tricher)
    drop += [c for c in DROP_COLS if c in df.columns] #supprimes les colonnes inutiles (bruits , overfitting,..)
    return [c for c in df.columns if c not in set(drop)] #colonnes finales 


def fit_numeric_preprocessor(df):
    df.columns = df.columns.str.strip() #supprimes les espaces
    candidate_cols = clean_column_list(df) #appel a la fonction prec
    numeric_cols = [] #les colonnes numeriques
    dropped_cols = [] #les colonnes supprimees

    for col in progress(candidate_cols, desc="Selection colonnes", leave=True):
        if is_numeric_dtype(df[col]): #verifie si la colonne est numerique
            numeric_cols.append(col)
            continue #si oui on passe a la suivante
        sample = df[col].dropna().head(20000) #si non on prend les valeurs Non nulles + Max 20000 lignes
        converted = pd.to_numeric(sample, errors="coerce") #si valeur invalide => NaN
        if len(sample) and converted.notna().mean() >= 0.95:
            numeric_cols.append(col) #numerique => acceptee
        else:
            dropped_cols.append(col) #rejetee 

    if dropped_cols:
        print("Colonnes non numeriques supprimees :", len(dropped_cols))
    if not numeric_cols:
        raise ValueError("Aucune colonne numerique exploitable.")

    arrays = []
    medians = {}
    feature_columns = []

    print("\nConstruction X fit colonne par colonne...")
    for col in progress(numeric_cols, desc="Fit numeric", leave=True):
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=True) #conversion en Float32(plus rapide, moins de memoire)
        arr[~np.isfinite(arr)] = np.nan #valeurs invalides + infinies => NaN
        if np.all(np.isnan(arr)):
            continue #si tout le colonnes est invalide on ignore 
        median = float(np.nanmedian(arr)) # calcule median
        if not np.isfinite(median):
            median = 0.0 #si median=NaN ou infinie alors => 0
        arr = np.nan_to_num(arr, nan=median, posinf=median, neginf=median).astype(np.float32, copy=False) #remplace inf + NaN par la median
        if float(np.min(arr)) == float(np.max(arr)):
            continue #si colonnes constante => supprimee
        #Sauvegarde
        medians[col] = median
        feature_columns.append(col)
        arrays.append(arr)

    if not arrays:
        raise ValueError("Toutes les colonnes numeriques sont constantes ou invalides.") #verifier s'il existe des colonnes valides

    X_raw = np.column_stack(arrays).astype(np.float32, copy=False) #Fusion des colonnes
    scaler = StandardScaler() #Standardisation des features
    X = scaler.fit_transform(X_raw).astype(np.float32)#Apprend Moy+Variance => transforme les donnees 
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) #si inf+NaN apparu => transformer a 0

    preprocessor = {
        "feature_columns": feature_columns, #features finales
        "medians": medians, #remplacer les NaN par le mm valeur
        "scaler": scaler, #contient Moy Variance + Param normalisation
        "dropped_cols": dropped_cols, #colonnes supprimee
    }
    return X, preprocessor #X=> Matrice prete pour PyTorch , Pipeline sauvegardee


def transform_numeric_preprocessor(df, preprocessor):
    df.columns = df.columns.str.strip()
    #Recuperation du PreProcessing appris
    feature_columns = preprocessor["feature_columns"]
    medians = preprocessor["medians"]
    scaler = preprocessor["scaler"]

    arrays = []
    missing = []

    print("\nTransformation X colonne par colonne...")
    for col in progress(feature_columns, desc="Transform numeric", leave=True):
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=True) #conversion en numerique + NaN pour col invalides
        else:
            #Si une colonne manque, on cree une pleine de NaN et on remplace avec
            arr = np.full(len(df), np.nan, dtype=np.float32)
            missing.append(col)
        arr[~np.isfinite(arr)] = np.nan # inf+Val invalide => NaN
        median = float(medians.get(col, 0.0)) #remplacer valeurs manquantes avec median
        arr = np.nan_to_num(arr, nan=median, posinf=median, neginf=median).astype(np.float32, copy=False)
        arrays.append(arr) #col ajoutee a la matrice

    if missing:
        print("Colonnes manquantes ajoutees :", len(missing))

    X_raw = np.column_stack(arrays).astype(np.float32, copy=False) #reconstruction de la matrice
    X = scaler.transform(X_raw).astype(np.float32) #application du Scaler
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) #Nettoyage


# =========================================================
# 4. OUTILS CLIENTS / HFL
# =========================================================
def normalize_proportions(proportions, num_clients):
    proportions = np.array(proportions, dtype=np.float64) #transforme la liste en tableau NumPy (float64 pour eviter les erreurs et + precisions)
    if len(proportions) != num_clients: #1 proportion par client
        raise ValueError("CLIENT_DATA_PROPORTIONS doit avoir la meme taille que NUM_CLIENTS.")
    if np.any(proportions <= 0): #positivite
        raise ValueError("Toutes les proportions doivent etre positives.")
    return proportions / proportions.sum()


def split_clients_unsupervised(X, num_clients, proportions): #X : matrice des donnees
    proportions = normalize_proportions(proportions, num_clients) #somme des proportions = 1
    indices = np.random.permutation(len(X)) #creer une permutation aleatoire
    split_points = (np.cumsum(proportions)[:-1] * len(X)).astype(int) #somme cumulative => supp dernier element => MUL par taille du dataset => Conversion INT
    client_indices = np.split(indices, split_points) #indice pour chaque proportion
    return [X[idx] for idx in client_indices]


def split_clients_supervised(X, y, num_clients, proportions): #X : Features , Y : Labels
    proportions = normalize_proportions(proportions, num_clients) #somme des proportions = 1
    indices = np.random.permutation(len(X)) #Creer une permutation aleatoire
    split_points = (np.cumsum(proportions)[:-1] * len(X)).astype(int) #somme cumulative => supp dernier element => MUL par taille du dataset => Conversion INT
    client_indices = np.split(indices, split_points)  #indice pour chaque proportion
    return [(X[idx], y[idx]) for idx in client_indices]


def weighted_average_state_dicts(state_dicts, sizes): #state_dicts : listes des modeles des clients , size : nbr data chaque client
    total_size = sum(sizes)
    #empeche la division par 0 + agregation invalide
    if total_size == 0:
        raise ValueError("Impossible d'agreger : total_size = 0.")

    avg_state = copy.deepcopy(state_dicts[0]) #creation d'une copie du modele
    for key in avg_state.keys():
        if not torch.is_floating_point(avg_state[key]): #si variable PyTorch n'est pas un poid flottant =>on les copie simplement
            avg_state[key] = state_dicts[0][key]
            continue
        avg_state[key] = torch.zeros_like(avg_state[key]) #creer un tenseur de 0
        for state, size in zip(state_dicts, sizes):
            avg_state[key] += state[key] * (size / total_size)
    return avg_state


def hfl_aggregate(local_weights, local_sizes, edge_groups, model_name="model"):
    print(f"\n===== AGREGATION HFL : {model_name} =====")
    edge_weights = [] #modele agrege a l'Edge
    edge_sizes = [] #taille de data par Edge

    for edge_idx, group in enumerate(edge_groups):
        group_weights = [local_weights[i] for i in group if local_sizes[i] > 0]#on recupere les clients actifs 
        group_sizes = [local_sizes[i] for i in group if local_sizes[i] > 0] #tailles des clients actifs
        if not group_weights:
            print(f"Edge {edge_idx + 1}: aucun client actif.") #si aucun client n'a participe on passe l'Edge
            continue

        edge_state = weighted_average_state_dicts(group_weights, group_sizes) #fusion a l'interieur du Edge (FedAvg)
        edge_size = sum(group_sizes) #taille totale des clients du Edge
        edge_weights.append(edge_state) #le resultat d'agregation de chaque Edge dans edge_weights(va comporter les resultats de tout les Edges)
        edge_sizes.append(edge_size) #la taille de chaque Edge dans edge_size
        print(f"Edge {edge_idx + 1} | clients={[i + 1 for i in group]} | size={edge_size}")

    global_state = weighted_average_state_dicts(edge_weights, edge_sizes) #agregations des Edges
    print("Serveur global | tailles edges :", edge_sizes)
    return global_state, edge_sizes


# =========================================================
# 5. MODELES
# =========================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim): #initialiser input_dim : nombre de features
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), #premiere couche de compression (ReLU : non linearite)
            nn.Linear(256, 128), nn.ReLU(), #deuxieme couche de compression
            nn.Linear(128, 16), nn.ReLU(), #derniere couche de compression
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(), #premiere couche de decompression
            nn.Linear(128, 256), nn.ReLU(), #deuxieme couche de decompression
            nn.Linear(256, input_dim), #troisieme couche de decompression
        )

    def forward(self, x): 
        return self.decoder(self.encoder(x)) #fonction qui fait le travail : x -> encode -> latent vector -> Decode -> x reconstruit

    def encode(self, x):
        return self.encoder(x) #retourne le vecteur Latent


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), #transformation des features a 256
            nn.BatchNorm1d(256), nn.Dropout(0.40), #batch : stabilise l'apprentissage , Dropout : desactive 40% des neurones (eviter l'overfitting)
            nn.Linear(256, 128), nn.ReLU(), #256 -> 128 
            nn.BatchNorm1d(128), nn.Dropout(0.30), # 30% des neurones desactivee
            nn.Linear(128, 64), nn.ReLU(), #128 -> 64
            nn.BatchNorm1d(64), nn.Dropout(0.20), # 20% des neurones sont desactivee
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x) #Lance l'operation input_dim -> 256 -> 128 -> 64 -> 2


class FocalLoss(nn.Module): # elle fait l'equilibre entre les classes(Benign , Attack) si ya une dominance elle equilibre en augmentant le loss 
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets): #logits : sorties brutes du modele , targets : vraies classes
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none") #calcule l'erreur classique par echantillon , ce = -log(pt)
        pt = torch.exp(-ce) # pt = e(-ce) 
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


# =========================================================
# 6. DATA LOADERS / TRAINING
# =========================================================
def make_ae_loader(X, shuffle=True): #preparer les donnees du train pour AE
    t = torch.from_numpy(np.asarray(X, dtype=np.float32)) #Numpy(listes) -> PyTorch(tensors)
    return DataLoader(TensorDataset(t, t), batch_size=BATCH_SIZE, shuffle=shuffle) #TensorDataSet(t,t) pour dire au AE que la reconstruction de donnes doit etre identique a l'entree
#Decoupage en Batchs + shuffle => melange aleatoirement

def make_supervised_loader(X, y, shuffle=True): #pour MLP prepare les donnees
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle) #X : Features , Y : Labels


def train_local_ae(global_model, client_X, client_id): #debut entrainement AE
    local_model = copy.deepcopy(global_model).to(DEVICE) #client recoit une copie des donnees poue eviter de modifier l'original
    local_model.train() #active le mode entrainement
    loader = make_ae_loader(client_X, shuffle=True) #cree les batchs
    criterion = nn.MSELoss() #calcule de la loss MOY((t(normal)-t(predit))^2)
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE_AE) #Adam met a jour les poids
    losses = []

    for epoch_idx in range(LOCAL_EPOCHS_AE): #entraine plusieurs fois(Epochs) son modele 
        running_loss = 0.0 #initialise la Loss
        iterator = progress(loader, desc=f"AE Client {client_id} E{epoch_idx + 1}/{LOCAL_EPOCHS_AE}", leave=False) #Barre de progression
        for batch_x, batch_target in iterator: #batch_x : entree , batch_target : cible
            #deplacer les Tensors vers la CPU ou GPU (si existante)
            batch_x = batch_x.to(DEVICE)
            batch_target = batch_target.to(DEVICE)
            optimizer.zero_grad() #remet les gradients a 0
            output = local_model(batch_x) #Ici l'AE effectue ses etapes : x -> encode -> decode ...
            loss = criterion(output, batch_target) # calcule de la Loss
            if torch.isnan(loss): #si la loss devient NaN ou instable on ignore le batch
                continue
            loss.backward() #calcule de gradients (ce qui indique comment modifier les poids pour reduire les erreurs) auto
            optimizer.step() #Adam modifie les poids
            running_loss += loss.item() * batch_x.size(0) #accumulation de perte totale
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}") #affichage de la loss en temps reel
        losses.append(running_loss / len(loader.dataset)) #La loss moyenne de l'epochs

    return local_model.state_dict(), len(client_X), float(np.mean(losses)) #Retour Final


def build_class_weights(y):
    class_counts = np.bincount(y, minlength=2) #calcule le poids de chaque classe 
    total = class_counts.sum()
    weights = total / (2.0 * np.maximum(class_counts, 1)) #donne plus de poids a la classe minoritaire
    weights[1] *= ATTACK_CLASS_WEIGHT_MULTIPLIER #augmente le poids de la classe attack
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_local_mlp(global_model, client_X, client_y, client_id):
    local_model = copy.deepcopy(global_model).to(DEVICE) #on donne une copie au client pour eviter de modifier l'original
    local_model.train() #actver le mode entrainement
    loader = make_supervised_loader(client_X, client_y, shuffle=True) #cree des mini batchs

    class_weights = build_class_weights(client_y) #calcule les poids des classes pour chaque client
    criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA) if USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=class_weights) #choisis la fonction de loss
    optimizer = optim.AdamW(local_model.parameters(), lr=LEARNING_RATE_MLP, weight_decay=WEIGHT_DECAY_MLP) #Mise a jour des poids
    losses = []

    for epoch_idx in range(LOCAL_EPOCHS_MLP): #entraine plusieur fois (epochs)  
        running_loss = 0.0 #initialise la loss
        iterator = progress(loader, desc=f"MLP Client {client_id} E{epoch_idx + 1}/{LOCAL_EPOCHS_MLP}", leave=False) #barre de progression
        for batch_x, batch_y in iterator: #donne des mini batchs
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad() #gradient a 0
            logits = local_model(batch_x) #produits des sorties brutes (Logits)
            loss = criterion(logits, batch_y) #calcule de la loss 
            if torch.isnan(loss): #si Loss NaN ou invalide passes le batch
                continue
            loss.backward() #calcule les gradients
            optimizer.step() #met a jour les poids
            running_loss += loss.item() * batch_x.size(0) #ajoute la perte de chaque batch multiplie par le nombre d'exemples
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
        losses.append(running_loss / len(loader.dataset)) #stock la perte moyenne de l'epoch

    return local_model.state_dict(), len(client_X), float(np.mean(losses)) #retourne les poids des modeles local , taille des donnees , la loss moyenne 


# =========================================================
# 7. AE FEATURES / SEUILS / EVALUATION
# =========================================================
def reconstruction_errors(model, X):
    model.eval() #met le modele en mode evaluation (a quel point l'AE reconstruit mal l'entree)
    loader = DataLoader(torch.from_numpy(np.asarray(X, dtype=np.float32)), batch_size=BATCH_SIZE, shuffle=False) #cree des batchs sans melanges (not train)
    errors = []
    with torch.no_grad(): #desactive le calcul des gradient par PyTorch
        for batch_x in loader:  
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x) #L'AE recnstruit les donness
            err = torch.mean((batch_x - output) ** 2, dim=1) #calcul l'erreur 
            errors.extend(err.cpu().numpy())
    return np.array(errors, dtype=np.float32) #retourne un tableau d'erreurs 

#on initialise le meilleur resulatat trouve en se basant sur ca on va choisir le seuil automatiquemnet
def choose_auto_threshold_percentile(normal_reference_errors, validation_errors, validation_y):
    best = {
        "percentile": None,
        "threshold": None,
        "accuracy": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
        "fpr": 1.0,
    }

    for percentile in AUTO_THRESHOLD_PERCENTILES: #test plusieurs seuils afin de choisir le meilleur
        threshold = float(np.percentile(normal_reference_errors, percentile)) #seuil calcule a partir des erreurs des donnees normales
        pred = (validation_errors > threshold).astype(np.int64) #cree les predictions
        #calcul des metrique pour evaluer le seuil
        acc = accuracy_score(validation_y, pred)
        prec = precision_score(validation_y, pred, zero_division=0)
        rec = recall_score(validation_y, pred, zero_division=0)
        f1 = f1_score(validation_y, pred, zero_division=0)  
        tn, fp, fn, tp = confusion_matrix(validation_y, pred, labels=[0, 1]).ravel() #recupere les FP,FN.TP,TN
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 #le taux de fausses alertes

        if f1 > best["f1"] or (np.isclose(f1, best["f1"]) and fpr < best["fpr"]): 
            #le code garde le seuil si F1 est meilleur ou si F1 est egal mais avec moins de fausses alertes
            best = {
                "percentile": float(percentile),
                "threshold": threshold,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "fpr": float(fpr),
            }
    return best


def extract_ae_features_and_detection(model, X, threshold, normal_error_std):
    model.eval() #passage au mode evaluation
    X_np = np.asarray(X, dtype=np.float32)
    loader = DataLoader(torch.from_numpy(X_np), batch_size=BATCH_SIZE, shuffle=False)

    # On determine latent_dim sans construire une enorme liste en memoire.
    with torch.no_grad():
        sample = torch.from_numpy(X_np[:1]).to(DEVICE)
        latent_dim = int(model.encode(sample).shape[1]) #determine la taille de sortie de l'encodeur = 16

    extra_dim = 0
    if USE_RECONSTRUCTION_ERROR_AS_FEATURE:
        extra_dim += 1 #ajoute l'erreur de reconstruction
    if USE_AE_DECISION_AS_MLP_FEATURE:
        extra_dim += 2 #ajoute la decision finale + score normalisee de l'AE 
    #donc logiquement X_mlp = 16 + 1 + 2 = 19 
    X_mlp = np.empty((len(X_np), latent_dim + extra_dim), dtype=np.float32) #cree une matrice finale pour le MLP
    err_all = np.empty(len(X_np), dtype=np.float32) #un tableau qui stocke l'erreur de reconstruction
    ae_pred_all = np.empty(len(X_np), dtype=np.int64) #un autre pour la prediction
    ae_score_all = np.empty(len(X_np), dtype=np.float32) #et un autre pour le score d'anomalie normalise

    start = 0
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE)
            z = model.encode(batch_x) #extrait le vecteur latent
            x_hat = model(batch_x) #reconstruction de l'entree
            err = torch.mean((batch_x - x_hat) ** 2, dim=1) #calcul de l'erreur de reconstruction pour chaque echantillion

            batch_size = batch_x.size(0) #la taille du batch actuel
            end = start + batch_size #determine ou se termine le batch actuel dans le tableau

            z_np = z.cpu().numpy().astype(np.float32, copy=False) #conversion du vector latent du PyTorch -> Numpy
            err_np = err.cpu().numpy().astype(np.float32, copy=False) #meme chose pour les erreurs de reconstruction
            pred_np = (err_np > threshold).astype(np.int64) #si l'erreur depasse le seuil -> attaque sinon -> normal
            score_np = ((err_np - threshold) / max(normal_error_std, 1e-12)).astype(np.float32) #a quel point l'erreur est loin du seuil

            col = 0
            X_mlp[start:end, col:col + latent_dim] = z_np #ajoute des features latentes dans l'entree du MLP
            col += latent_dim

            if USE_RECONSTRUCTION_ERROR_AS_FEATURE:
                X_mlp[start:end, col] = err_np #ajout de l'erreur de reconstruction
                col += 1

            if USE_AE_DECISION_AS_MLP_FEATURE:
                X_mlp[start:end, col] = pred_np.astype(np.float32) #les predictions aussi
                X_mlp[start:end, col + 1] = score_np #eventuellement le score

            err_all[start:end] = err_np
            ae_pred_all[start:end] = pred_np
            ae_score_all[start:end] = score_np
            start = end

    return X_mlp, err_all, ae_pred_all, ae_score_all


def predict_mlp(model, X):
    model.eval() #mode evaluation
    loader = DataLoader(torch.from_numpy(np.asarray(X, dtype=np.float32)), batch_size=BATCH_SIZE, shuffle=False) #decoupe les donnees en batchs
    preds = []
    probs_attack = []
    softmax = nn.Softmax(dim=1) #MLP produit des Logits

    with torch.no_grad(): #desactiver le comptage des gradients
        for batch_x in loader:
            batch_x = batch_x.to(DEVICE) #envoie a la CPU ou GPU (si elle existe)
            probs = softmax(model(batch_x)) #lance l'operation MLP batch_x -> MLP -> Logits -> softmax -> predictions
            preds.extend(torch.argmax(probs, dim=1).cpu().numpy()) #argmax choisis la classe avec la plus grande probabilites
            probs_attack.extend(probs[:, 1].cpu().numpy()) #extrait seulement les probabilites d'attaques

    return np.array(preds), np.array(probs_attack)


def choose_mlp_threshold_recall_fpr(y_true, attack_scores, max_fpr=0.08):
    best = None
    fallback = None

    for threshold in MLP_THRESHOLD_GRID: #test de plusieurs seuils 
        pred = (attack_scores >= threshold).astype(np.int64) #si probabilite >= seuil -> Attaque
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel() #recupere les FN,FP,TP,TN
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 #calcule le FPR
        #calcul des metriques
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=2.0, zero_division=0)
        acc = accuracy_score(y_true, pred)
        #sauvegarde les metriques
        info = {
            "threshold": float(threshold),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "f2": float(f2),
            "fpr": float(fpr),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        if fallback is None or info["f2"] > fallback["f2"]:
            fallback = info #garder le meilleur F2 score global mm si aucun seuil ne respecte la FPR

        if fpr <= max_fpr: #on impose 8% pour la FPR
            if best is None:
                best = info
            elif info["recall"] > best["recall"] or (np.isclose(info["recall"], best["recall"]) and info["f1"] > best["f1"]):
                best = info #maximiser le recall

    if best is None: #si aucun seuil ne respecte le FPR maximal alors on prends le meilleur F2
        fallback["note"] = "Aucun seuil ne respecte MAX_ALLOWED_FPR; seuil choisi par meilleur F2."
        return fallback
    best["note"] = f"Seuil choisi pour maximiser recall avec FPR <= {max_fpr}."
    return best


def evaluate(name, y_true, y_pred, scores=None):
    #calcul des metriques 
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) #construit la matrice de confusion
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 #taux de fausses alertes
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 #taux t'attaques ratees
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 #capacite a reconnaitre les vrais normaux

    try:
        auc = roc_auc_score(y_true, scores) if scores is not None else np.nan #calcul ROC-AUC
    except Exception:
        auc = np.nan
    #affiche tout les resutats
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
    print("\nMatrice de confusion :")
    print(cm)
    print("\nClassification report :")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=4, zero_division=0))

    #retourner toutes les metrique dans un dictionnaire python
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================================================
# 8. CHARGEMENT BENIGN POUR AE
# =========================================================
df_benign, benign_sources = load_folder_with_labels(BENIGN_FOLDER, infer_labels=False) #chargement des fichiers du trafic normal
print("\n===== BENIGN POUR AUTOENCODEUR =====")
print("Shape brute benign :", df_benign.shape)

X_benign, preprocessor = fit_numeric_preprocessor(df_benign) #nettoie les donnees (pre-traitement)
feature_columns = preprocessor["feature_columns"] #recupere les colonnes reelement utilisee

print("Shape benign apres pretraitement :", X_benign.shape)
print("Nombre de features :", len(feature_columns))

#separes le BENIGN en train (80%) + Validation(20%) 
X_ae_train, X_ae_val = train_test_split(
    X_benign,
    test_size=VALID_SIZE_AE,
    random_state=RANDOM_STATE,
)

print("\n===== SPLIT AUTOENCODEUR =====")
print("AE train :", X_ae_train.shape)
print("AE val   :", X_ae_val.shape)

clients_ae = split_clients_unsupervised(X_ae_train, NUM_CLIENTS, CLIENT_DATA_PROPORTIONS) #repartis les donnees aux clients
print("\n===== CLIENTS AE HFL NON EQUITABLES =====")
for i, client_X in enumerate(clients_ae):
    print(f"Client AE {i + 1}: {client_X.shape}")


# =========================================================
# 9. ENTRAINEMENT HFL + AE
# =========================================================
input_dim = X_benign.shape[1]
ae_model = Autoencoder(input_dim).to(DEVICE) #cree l'AE avec le bon nombre de features

#prepare la sauvegarde du meilleur AE
best_ae_val_loss = float("inf")
best_ae_path = os.path.join(SAVE_FOLDER, "best_hfl_autoencoder.pth")

print("\n===== ENTRAINEMENT HFL + AUTOENCODEUR =====")

for round_idx in range(GLOBAL_ROUNDS):
    print(f"\n--- Round global AE {round_idx + 1}/{GLOBAL_ROUNDS} ---")
    local_weights = []
    local_sizes = []
    local_losses = []
    #dans cette boucle : Clients entrainent localement -> Edge agrege -> serveur global agrege -> Validation
    for client_idx, client_X in enumerate(clients_ae):
        weights, size, loss = train_local_ae(ae_model, client_X, client_idx + 1) #client recoit le modele globale entraine puis retourne : poids , taille , loss
        local_weights.append(weights)
        local_sizes.append(size)
        local_losses.append(loss)
        print(f"Client AE {client_idx + 1} | size={size} | loss={loss:.6f}")

    new_weights, edge_sizes = hfl_aggregate(local_weights, local_sizes, EDGE_GROUPS, model_name="Autoencoder") #fusion hierarchique client -> Edge -> serveur  
    ae_model.load_state_dict(new_weights) #mets a jour l'AE global avec les poids agrege

    val_errors = reconstruction_errors(ae_model, X_ae_val) #evalue l'AE
    val_loss = float(np.mean(val_errors))

    print("Loss moyenne clients AE :", np.mean(local_losses))
    print("Validation loss AE      :", val_loss)

    if val_loss < best_ae_val_loss: #si le modele est meilleur que tout les precedent il le sauvegarde
        best_ae_val_loss = val_loss
        torch.save(ae_model.state_dict(), best_ae_path)
        print(">> Meilleur autoencodeur HFL sauvegarde.")

best_ae = Autoencoder(input_dim).to(DEVICE)
best_ae.load_state_dict(torch.load(best_ae_path, map_location=DEVICE)) #on recharge le meilleur AE trouvee 
best_ae.eval() #on le mets a l'evaluation


# =========================================================
# 10. CHARGEMENT DATASET MIXTE + SPLITS
# =========================================================
df_mixed, y_mixed, source_files = load_folder_with_labels(MIXED_FOLDER, infer_labels=True) #charger les CSV du dossier mixte(Attack,Benign)

print("\n===== DATASET MIXTE =====")
print("Shape brute mixed :", df_mixed.shape)
print("Total     :", len(y_mixed))
print("Normaux   :", np.sum(y_mixed == 0))
print("Attaques  :", np.sum(y_mixed == 1))

X_mixed = transform_numeric_preprocessor(df_mixed, preprocessor) #Pre-traitement
print("Shape mixed apres pretraitement :", X_mixed.shape)

#divises en train 70% et test 30%
X_train_raw, X_test_raw, y_train, y_test, src_train, src_test = train_test_split(
    X_mixed,
    y_mixed,
    source_files,
    test_size=MLP_TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_mixed,
)

#separer le Train en Train 80% et Validation 20%
X_mlp_train_raw, X_mlp_val_raw, y_mlp_train, y_mlp_val = train_test_split(
    X_train_raw,
    y_train,
    test_size=MLP_VALID_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

print("\n===== SPLIT MLP =====")
print("MLP train :", X_mlp_train_raw.shape)
print("MLP val   :", X_mlp_val_raw.shape)
print("MLP test  :", X_test_raw.shape)
print("Train benign :", np.sum(y_mlp_train == 0), "| Train attack :", np.sum(y_mlp_train == 1))
print("Val benign   :", np.sum(y_mlp_val == 0), "| Val attack   :", np.sum(y_mlp_val == 1))
print("Test benign  :", np.sum(y_test == 0), "| Test attack  :", np.sum(y_test == 1))


# =========================================================
# 11. SEUIL AE AUTOMATIQUE + FEATURES AE POUR MLP
# =========================================================
print("\n===== CHOIX AUTOMATIQUE DU SEUIL AE =====")

normal_reference_errors = reconstruction_errors(best_ae, X_ae_train) #calcul des erreur de reconstruction sur le trafic normal
val_errors_for_threshold = reconstruction_errors(best_ae, X_mlp_val_raw) #calcul des erreurs 

#choix du meilleur seuil d'anomalie AE
ae_threshold_info = choose_auto_threshold_percentile(
    normal_reference_errors=normal_reference_errors,
    validation_errors=val_errors_for_threshold,
    validation_y=y_mlp_val,
)

AE_THRESHOLD_PERCENTILE = ae_threshold_info["percentile"]
AE_THRESHOLD = ae_threshold_info["threshold"]
AE_NORMAL_ERROR_STD = float(np.std(normal_reference_errors)) #dispersion des erreurs normales

print("threshold_percentile choisi :", AE_THRESHOLD_PERCENTILE)
print("threshold AE                :", AE_THRESHOLD)
print("Validation AE Accuracy      :", ae_threshold_info["accuracy"])
print("Validation AE Precision     :", ae_threshold_info["precision"])
print("Validation AE Recall        :", ae_threshold_info["recall"])
print("Validation AE F1            :", ae_threshold_info["f1"])
print("Validation AE FPR           :", ae_threshold_info["fpr"])

print("\n===== EXTRACTION FEATURES AE POUR MLP =====")
#"extract_ae_features_and_detection" -> extrait : Features latents, erreurs reconstruction, decision (attack/benign), Score AE
X_mlp_train, train_errors, train_ae_pred, train_ae_score = extract_ae_features_and_detection(
    best_ae, X_mlp_train_raw, AE_THRESHOLD, AE_NORMAL_ERROR_STD #x_mlpx_train -> Train
)
X_mlp_val, val_errors, val_ae_pred, val_ae_score = extract_ae_features_and_detection(
    best_ae, X_mlp_val_raw, AE_THRESHOLD, AE_NORMAL_ERROR_STD #x_mlp_val -> Validation
)
X_test_mlp, test_errors, test_ae_pred, test_ae_score = extract_ae_features_and_detection(
    best_ae, X_test_raw, AE_THRESHOLD, AE_NORMAL_ERROR_STD #x_test_mlp -> Test
)

print("Train AE F1 :", f1_score(y_mlp_train, train_ae_pred, zero_division=0))
print("Val AE F1   :", f1_score(y_mlp_val, val_ae_pred, zero_division=0))
print("Val AE confusion :")
print(confusion_matrix(y_mlp_val, val_ae_pred, labels=[0, 1]))

mlp_feature_scaler = StandardScaler() #standardise tout les features du MLP parceque avant les features ont des echelles differentes
X_mlp_train = mlp_feature_scaler.fit_transform(X_mlp_train).astype(np.float32)
X_mlp_val = mlp_feature_scaler.transform(X_mlp_val).astype(np.float32)
X_test_mlp = mlp_feature_scaler.transform(X_test_mlp).astype(np.float32)

print("X_mlp_train :", X_mlp_train.shape)
print("X_mlp_val   :", X_mlp_val.shape)
print("X_test_mlp  :", X_test_mlp.shape)


# =========================================================
# 12. CLIENTS + ENTRAINEMENT HFL MLP
# =========================================================
clients_mlp = split_clients_supervised(X_mlp_train, y_mlp_train, NUM_CLIENTS, CLIENT_DATA_PROPORTIONS) #repartition des donnees X_client,Y_client

print("\n===== CLIENTS MLP HFL NON EQUITABLES =====")
for i, (client_X, client_y) in enumerate(clients_mlp):
    print(f"Client MLP {i + 1}: X={client_X.shape} | benign={np.sum(client_y == 0)} | attack={np.sum(client_y == 1)}")

mlp_input_dim = X_mlp_train.shape[1] #le MLP prends comme entree les resultats de l'AE
mlp_model = MLPBinaryClassifier(mlp_input_dim).to(DEVICE)

best_mlp_val_f2 = -1.0
best_mlp_path = os.path.join(SAVE_FOLDER, "best_hfl_mlp_binary.pth")

print("\n===== ENTRAINEMENT HFL + MLP BINAIRE =====")

for round_idx in range(GLOBAL_ROUNDS):
    print(f"\n--- Round global MLP {round_idx + 1}/{GLOBAL_ROUNDS} ---")
    local_weights = []
    local_sizes = []
    local_losses = []

    for client_idx, (client_X, client_y) in enumerate(clients_mlp):
        weights, size, loss = train_local_mlp(mlp_model, client_X, client_y, client_idx + 1) #chaque client retourne : poids, taille data , loss moyenne
        local_weights.append(weights)
        local_sizes.append(size)
        local_losses.append(loss)
        print(f"Client MLP {client_idx + 1} | size={size} | loss={loss:.6f}")

    new_weights, edge_sizes = hfl_aggregate(local_weights, local_sizes, EDGE_GROUPS, model_name="MLP") #fusionne les modele clients : client -> Edge -> server
    mlp_model.load_state_dict(new_weights)

    val_pred_argmax, val_scores = predict_mlp(mlp_model, X_mlp_val) #prediction argmax et score/probabilite Attack
    val_threshold_info = choose_mlp_threshold_recall_fpr(y_mlp_val, val_scores, MAX_ALLOWED_FPR) #meilleur seuil pour maximiser le recall et FPR<= 8%
    val_pred_threshold = (val_scores >= val_threshold_info["threshold"]).astype(np.int64) #decision finale depends du seuil
    val_f2 = fbeta_score(y_mlp_val, val_pred_threshold, beta=2.0, zero_division=0) #calcule le F2 score 
    val_f1 = f1_score(y_mlp_val, val_pred_threshold, zero_division=0) #calcule le F1 score
    val_acc = accuracy_score(y_mlp_val, val_pred_threshold) #calcule l'accuracy

    print("Loss moyenne clients MLP :", np.mean(local_losses))
    print("Validation Accuracy seuil:", val_acc)
    print("Validation F1 seuil      :", val_f1)
    print("Validation F2 seuil      :", val_f2)
    print("Validation Recall seuil  :", val_threshold_info["recall"])
    print("Validation FPR seuil     :", val_threshold_info["fpr"])
    print("Seuil MLP temporaire     :", val_threshold_info["threshold"])

    if val_f2 > best_mlp_val_f2: #choix selon le F2 scores (parceque le F2 score donne plus d'importance au recall)
        best_mlp_val_f2 = val_f2
        torch.save(mlp_model.state_dict(), best_mlp_path)
        print(">> Meilleur MLP HFL sauvegarde.")

#recharger le meilleur MLP trouve
best_mlp = MLPBinaryClassifier(mlp_input_dim).to(DEVICE)
best_mlp.load_state_dict(torch.load(best_mlp_path, map_location=DEVICE))
best_mlp.eval()


# =========================================================
# 13. SEUIL FINAL MLP + TEST
# =========================================================
print("\n===== CHOIX FINAL DU SEUIL MLP RECALL/FPR =====")
_, val_scores = predict_mlp(best_mlp, X_mlp_val) #meilleur MLP produit
mlp_threshold_info = choose_mlp_threshold_recall_fpr(y_mlp_val, val_scores, MAX_ALLOWED_FPR) #cherche meilleur seuil final -> recall maximal
MLP_DECISION_THRESHOLD = mlp_threshold_info["threshold"] #devient le seuil officiel du systeme IDS

print("Seuil MLP choisi :", MLP_DECISION_THRESHOLD)
print("Validation MLP Accuracy :", mlp_threshold_info["accuracy"])
print("Validation MLP Precision:", mlp_threshold_info["precision"])
print("Validation MLP Recall   :", mlp_threshold_info["recall"])
print("Validation MLP F1       :", mlp_threshold_info["f1"])
print("Validation MLP F2       :", mlp_threshold_info["f2"])
print("Validation MLP FPR      :", mlp_threshold_info["fpr"])
print("Note seuil              :", mlp_threshold_info["note"])

print("\n===== TEST FINAL SUR TEST SET =====")

#test sur data set jamais vu
test_pred_argmax, test_scores = predict_mlp(best_mlp, X_test_mlp) #choisi la classe la plus probable
test_pred_threshold = (test_scores >= MLP_DECISION_THRESHOLD).astype(np.int64) #seuil optimise

ae_results = evaluate("HFL + AUTOENCODEUR SEUL - SEUIL AUTO", y_test, test_ae_pred, scores=test_errors) #evalue l'AE seul
mlp_argmax_results = evaluate("HFL AE-FEATURES -> MLP ARGMAX", y_test, test_pred_argmax, scores=test_scores) #MLP decision classique
mlp_threshold_results = evaluate("HFL AE-FEATURES -> MLP SEUIL RECALL/FPR", y_test, test_pred_threshold, scores=test_scores) #evalue l'IDS final 


# =========================================================
# 14. EVALUATION OPTIONNELLE SUR TOUT LE DOSSIER MIXTE
# =========================================================
print("\n===== EVALUATION OPTIONNELLE SUR TOUT LE DOSSIER MIXTE =====")
X_all_mlp, all_errors, all_ae_pred, all_ae_score = extract_ae_features_and_detection(
    best_ae, X_mixed, AE_THRESHOLD, AE_NORMAL_ERROR_STD
)
X_all_mlp = mlp_feature_scaler.transform(X_all_mlp).astype(np.float32)
all_pred_argmax, all_scores = predict_mlp(best_mlp, X_all_mlp)
all_pred_threshold = (all_scores >= MLP_DECISION_THRESHOLD).astype(np.int64)

all_ae_results = evaluate("ALL - HFL + AUTOENCODEUR SEUL", y_mixed, all_ae_pred, scores=all_errors)
all_mlp_results = evaluate("ALL - HFL AE-FEATURES -> MLP SEUIL RECALL/FPR", y_mixed, all_pred_threshold, scores=all_scores)


# =========================================================
# 15. TABLEAU COMPARATIF
# =========================================================
comparison_df = pd.DataFrame({
    "Model": [
        "Test AE seuil auto",
        "Test MLP argmax",
        "Test MLP seuil Recall/FPR",
        "All AE seuil auto",
        "All MLP seuil Recall/FPR",
    ],
    "Accuracy": [
        ae_results["accuracy"],
        mlp_argmax_results["accuracy"],
        mlp_threshold_results["accuracy"],
        all_ae_results["accuracy"],
        all_mlp_results["accuracy"],
    ],
    "Precision": [
        ae_results["precision"],
        mlp_argmax_results["precision"],
        mlp_threshold_results["precision"],
        all_ae_results["precision"],
        all_mlp_results["precision"],
    ],
    "Recall": [
        ae_results["recall"],
        mlp_argmax_results["recall"],
        mlp_threshold_results["recall"],
        all_ae_results["recall"],
        all_mlp_results["recall"],
    ],
    "F1-score": [
        ae_results["f1_score"],
        mlp_argmax_results["f1_score"],
        mlp_threshold_results["f1_score"],
        all_ae_results["f1_score"],
        all_mlp_results["f1_score"],
    ],
    "F2-score": [
        ae_results["f2_score"],
        mlp_argmax_results["f2_score"],
        mlp_threshold_results["f2_score"],
        all_ae_results["f2_score"],
        all_mlp_results["f2_score"],
    ],
    "FPR": [
        ae_results["fpr"],
        mlp_argmax_results["fpr"],
        mlp_threshold_results["fpr"],
        all_ae_results["fpr"],
        all_mlp_results["fpr"],
    ],
    "FNR": [
        ae_results["fnr"],
        mlp_argmax_results["fnr"],
        mlp_threshold_results["fnr"],
        all_ae_results["fnr"],
        all_mlp_results["fnr"],
    ],
    "ROC-AUC": [
        ae_results["roc_auc"],
        mlp_argmax_results["roc_auc"],
        mlp_threshold_results["roc_auc"],
        all_ae_results["roc_auc"],
        all_mlp_results["roc_auc"],
    ],
})

print("\n===== COMPARAISON FINALE =====")
print(comparison_df)


# =========================================================
# 16. SAUVEGARDE
# =========================================================
torch.save(best_ae.state_dict(), os.path.join(SAVE_FOLDER, "hfl_autoencoder_final.pth"))
torch.save(best_mlp.state_dict(), os.path.join(SAVE_FOLDER, "hfl_mlp_binary_final.pth"))

joblib.dump(preprocessor, os.path.join(SAVE_FOLDER, "preprocessor.pkl"))
joblib.dump(mlp_feature_scaler, os.path.join(SAVE_FOLDER, "mlp_feature_scaler.pkl"))
joblib.dump(feature_columns, os.path.join(SAVE_FOLDER, "feature_columns.pkl"))
joblib.dump(ae_threshold_info, os.path.join(SAVE_FOLDER, "ae_threshold_info.pkl"))
joblib.dump(MLP_DECISION_THRESHOLD, os.path.join(SAVE_FOLDER, "mlp_decision_threshold.pkl"))
joblib.dump(mlp_threshold_info, os.path.join(SAVE_FOLDER, "mlp_threshold_info.pkl"))

metadata = {
    "architecture": (
        "Optimized HFL + AE + MLP for CICIoT. AE trained on benign only. "
        "AE threshold auto-selected on labeled validation. MLP receives AE latent features, "
        "reconstruction error, AE decision and AE score. MLP uses HFL, FocalLoss, attack weighting, "
        "AdamW and final recall/FPR decision threshold."
    ),
    "num_clients": NUM_CLIENTS,
    "num_edges": NUM_EDGES,
    "edge_groups": EDGE_GROUPS,
    "client_data_proportions": CLIENT_DATA_PROPORTIONS,
    "global_rounds": GLOBAL_ROUNDS,
    "local_epochs_ae": LOCAL_EPOCHS_AE,
    "local_epochs_mlp": LOCAL_EPOCHS_MLP,
    "batch_size": BATCH_SIZE,
    "learning_rate_ae": LEARNING_RATE_AE,
    "learning_rate_mlp": LEARNING_RATE_MLP,
    "weight_decay_mlp": WEIGHT_DECAY_MLP,
    "use_focal_loss": USE_FOCAL_LOSS,
    "focal_gamma": FOCAL_GAMMA,
    "attack_class_weight_multiplier": ATTACK_CLASS_WEIGHT_MULTIPLIER,
    "max_allowed_fpr": MAX_ALLOWED_FPR,
    "best_ae_val_loss": best_ae_val_loss,
    "best_mlp_val_f2": best_mlp_val_f2,
    "input_dim": input_dim,
    "mlp_input_dim": mlp_input_dim,
    "ae_threshold_percentile": AE_THRESHOLD_PERCENTILE,
    "ae_threshold": AE_THRESHOLD,
    "mlp_decision_threshold": MLP_DECISION_THRESHOLD,
    "use_reconstruction_error_as_feature": USE_RECONSTRUCTION_ERROR_AS_FEATURE,
    "use_ae_decision_as_mlp_feature": USE_AE_DECISION_AS_MLP_FEATURE,
}

joblib.dump(metadata, os.path.join(SAVE_FOLDER, "metadata_hfl_ae_mlp_optimized.pkl"))
joblib.dump({
    "test_ae_results": ae_results,
    "test_mlp_argmax_results": mlp_argmax_results,
    "test_mlp_threshold_results": mlp_threshold_results,
    "all_ae_results": all_ae_results,
    "all_mlp_threshold_results": all_mlp_results,
    "ae_threshold_info": ae_threshold_info,
    "mlp_threshold_info": mlp_threshold_info,
}, os.path.join(SAVE_FOLDER, "metrics_hfl_ae_mlp_optimized.pkl"))

comparison_df.to_csv(os.path.join(SAVE_FOLDER, "comparison_results_hfl_ae_mlp_optimized.csv"), index=False)

test_results_df = pd.DataFrame({
    "source_file": src_test,
    "y_true": y_test,
    "ae_pred": test_ae_pred,
    "mlp_pred_argmax": test_pred_argmax,
    "mlp_pred_threshold": test_pred_threshold,
    "ae_reconstruction_error": test_errors,
    "ae_score": test_ae_score,
    "mlp_attack_score": test_scores,
})
test_results_path = os.path.join(SAVE_FOLDER, "detailed_predictions_test_hfl_ae_mlp_optimized.csv")
test_results_df.to_csv(test_results_path, index=False)

all_results_df = pd.DataFrame({
    "source_file": source_files,
    "y_true": y_mixed,
    "ae_pred": all_ae_pred,
    "mlp_pred_argmax": all_pred_argmax,
    "mlp_pred_threshold": all_pred_threshold,
    "ae_reconstruction_error": all_errors,
    "ae_score": all_ae_score,
    "mlp_attack_score": all_scores,
})
all_results_path = os.path.join(SAVE_FOLDER, "detailed_predictions_all_hfl_ae_mlp_optimized.csv")
all_results_df.to_csv(all_results_path, index=False)

print("\n===== SAUVEGARDE TERMINEE =====")
print("Dossier resultats :", SAVE_FOLDER)
print("Fichier comparaison :", os.path.join(SAVE_FOLDER, "comparison_results_hfl_ae_mlp_optimized.csv"))
print("Fichier predictions test :", test_results_path)
print("Fichier predictions all  :", all_results_path)


