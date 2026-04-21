"""
IDS Federated Learning sur le dataset TON_IoT — Version 2 (corrigée)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Corrections v2 :
  1. FedAvg corrigé : poids partagés via numpy directement (pas MLPClassifier)
  2. Isolation Forest : contamination adaptative + threshold calibré sur train
  3. Évaluation propre : même test set pour tous les modèles

Usage :
  Télécharger le dataset TON_IoT depuis :
  https://research.unsw.edu.au/projects/toniot-datasets
  puis mettre le chemin du CSV dans DATASET_PATH en bas du fichier.
"""
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

'''
# Fusionner tous les CSV du dossier Network
fichiers = glob.glob("C:\\Users\\User\\Downloads\\Processed_datasets\\Processed_Network_dataset\\*.csv")
print(f"[INFO] {len(fichiers)} fichiers trouvés")

df = pd.concat([pd.read_csv(f) for f in fichiers], ignore_index=True)
df.to_csv("Network_complet.csv", index=False)
print(f"[INFO] Fichier fusionné : {df.shape}")
'''

# ════════════════════════════════════════════
# 1. CHARGEMENT & PRÉTRAITEMENT
# ════════════════════════════════════════════

def load_and_preprocess(filepath: str):
    print(f"[INFO] Chargement : {filepath}")
    if not __import__('os').path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERREUR] Fichier introuvable : '{filepath}'\n"
            f"\nVeuillez télécharger le dataset TON_IoT ici :\n"
            f"  https://research.unsw.edu.au/projects/toniot-datasets\n"
            f"\nPuis mettez le bon chemin dans la variable DATASET_PATH "
            f"en bas du fichier.\n"
            f"Exemple : DATASET_PATH = 'C:/Users/moi/Downloads/Train_Test_Network.csv'\n"
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Nettoyage
    cols_to_drop = ['ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'type']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col != 'label':
            df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=['label']).values
    y = (df['label'] != 0).astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[INFO] Normal : {(y==0).sum()} | Attaque : {(y==1).sum()}")
    print(f"[INFO] Taux d'attaque : {y.mean()*100:.1f}%\n")
    return X, y



# ════════════════════════════════════════════
# 2. RÉSEAU DE NEURONES MANUEL (NumPy pur)
#    → Compatible avec vrai FedAvg
# ════════════════════════════════════════════

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)


class NeuralNet:
    """
    MLP 2 couches cachées implémenté en NumPy pur.
    Les poids sont de simples arrays numpy → FedAvg natif.
    Architecture : input → 64 → 32 → 1 (sigmoid)
    """
    def __init__(self, n_features: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Initialisation He (adapté ReLU)
        self.W1 = rng.randn(n_features, 64) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(64)
        self.W2 = rng.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(32)
        self.W3 = rng.randn(32, 1)  * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(1)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = sigmoid(self.z3)
        return self.a3

    def predict_proba(self, X):
        return self.forward(X).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_loss(self, X, y):
        """Binary cross-entropy."""
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(proba) + (1-y) * np.log(1-proba))

    def train(self, X, y, lr=0.01, epochs=10, batch_size=64):
        """Mini-batch SGD avec backpropagation."""
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]

            for start in range(0, n, batch_size):
                Xb = X_s[start:start+batch_size]
                yb = y_s[start:start+batch_size].reshape(-1, 1)

                # Forward
                z1 = Xb @ self.W1 + self.b1
                a1 = relu(z1)
                z2 = a1 @ self.W2 + self.b2
                a2 = relu(z2)
                z3 = a2 @ self.W3 + self.b3
                a3 = sigmoid(z3)

                # Backward
                m  = len(Xb)
                d3 = (a3 - yb) / m                          # (m,1)
                dW3 = a2.T @ d3
                db3 = d3.sum(axis=0)

                d2 = (d3 @ self.W3.T) * relu_grad(z2)      # (m,32)
                dW2 = a1.T @ d2
                db2 = d2.sum(axis=0)

                d1 = (d2 @ self.W2.T) * relu_grad(z1)      # (m,64)
                dW1 = Xb.T @ d1
                db1 = d1.sum(axis=0)

                # Update
                self.W3 -= lr * dW3;  self.b3 -= lr * db3
                self.W2 -= lr * dW2;  self.b2 -= lr * db2
                self.W1 -= lr * dW1;  self.b1 -= lr * db1

    def get_weights(self):
        return [self.W1.copy(), self.b1.copy(),
                self.W2.copy(), self.b2.copy(),
                self.W3.copy(), self.b3.copy()]

    def set_weights(self, weights):
        self.W1, self.b1 = weights[0].copy(), weights[1].copy()
        self.W2, self.b2 = weights[2].copy(), weights[3].copy()
        self.W3, self.b3 = weights[4].copy(), weights[5].copy()


# ════════════════════════════════════════════
# 3. FEDAVG — Moyenne pondérée des poids
# ════════════════════════════════════════════

def federated_average(client_weights_list, client_sizes):
    """
    FedAvg : w_global = Σ (n_k / n_total) × w_k
    """
    total = sum(client_sizes)
    n_layers = len(client_weights_list[0])
    averaged = []
    for layer_idx in range(n_layers):
        layer_avg = sum(
            client_weights_list[i][layer_idx] * (client_sizes[i] / total)
            for i in range(len(client_weights_list))
        )
        averaged.append(layer_avg)
    return averaged


# ════════════════════════════════════════════
# 4. DISTRIBUTION DES DONNÉES
# ════════════════════════════════════════════

def split_data_among_clients(X, y, n_clients: int, iid: bool = True):
    clients_data = []
    if iid:
        idx = np.random.permutation(len(X))
        for split in np.array_split(idx, n_clients):
            clients_data.append((X[split], y[split]))
    else:
        # Non-IID : taux d'attaque différent par client
        att_idx = np.where(y == 1)[0]; np.random.shuffle(att_idx)
        nor_idx = np.where(y == 0)[0]; np.random.shuffle(nor_idx)
        att_splits = np.array_split(att_idx, n_clients)
        nor_splits = np.array_split(nor_idx, n_clients)
        ratios = np.linspace(0.1, 0.8, n_clients)
        for i in range(n_clients):
            n_att = int(len(att_splits[i]) * ratios[i])
            idx = np.concatenate([att_splits[i][:n_att], nor_splits[i]])
            np.random.shuffle(idx)
            clients_data.append((X[idx], y[idx]))

    print(f"[INFO] Répartition {'IID' if iid else 'Non-IID'} :")
    for i, (Xc, yc) in enumerate(clients_data):
        print(f"  Client {i+1:02d} : {len(Xc):5d} samples | "
              f"attaques = {yc.mean()*100:.1f}%")
    return clients_data


# ════════════════════════════════════════════
# 5. BOUCLE FEDERATED LEARNING
# ════════════════════════════════════════════

def federated_training(clients_data, X_test, y_test,
                       n_rounds=15, lr=0.005, local_epochs=8):
    print("\n" + "="*55)
    print("  FEDERATED LEARNING — MLP NumPy + FedAvg")
    print("="*55)

    n_features = clients_data[0][0].shape[1]

    # Initialiser le modèle global (même seed → même architecture)
    global_net = NeuralNet(n_features, seed=0)

    history = {'round': [], 'accuracy': [], 'f1': [], 'loss': []}

    for r in range(1, n_rounds + 1):
        client_weights = []
        client_sizes   = []
        client_losses  = []

        for i, (Xc, yc) in enumerate(clients_data):
            # Copier poids globaux → client
            local_net = NeuralNet(n_features, seed=i)
            local_net.set_weights(global_net.get_weights())

            # Entraînement local
            local_net.train(Xc, yc, lr=lr, epochs=local_epochs)

            loss = local_net.compute_loss(Xc, yc)
            client_losses.append(loss)
            client_weights.append(local_net.get_weights())
            client_sizes.append(len(Xc))

        # FedAvg
        global_weights = federated_average(client_weights, client_sizes)
        global_net.set_weights(global_weights)

        # Évaluation globale
        y_pred = global_net.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        avg_loss = np.mean(client_losses)

        history['round'].append(r)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['loss'].append(avg_loss)

        print(f"  Round {r:02d}/{n_rounds} | "
              f"acc={acc:.4f} | f1={f1:.4f} | loss={avg_loss:.4f}")

    # Rapport final
    y_pred_final = global_net.predict(X_test)
    print("\n[RAPPORT FINAL — MLP Fédéré]")
    print(classification_report(y_test, y_pred_final,
                                target_names=["Normal", "Attaque"]))
    return global_net, history, y_pred_final


# ════════════════════════════════════════════
# 6. ISOLATION FOREST FÉDÉRÉ (corrigé)
# ════════════════════════════════════════════

def federated_isolation_forest(clients_data, X_train, y_train,
                               X_test, y_test):
    """
    Correction : le threshold est calibré sur le train set
    pour correspondre au vrai taux d'attaque (pas un percentile fixe).
    """
    print("\n" + "="*55)
    print("  ISOLATION FOREST FÉDÉRÉ")
    print("="*55)

    # Contamination = taux réel d'attaques dans le train
    contamination = float(y_train.mean())
    contamination = np.clip(contamination, 0.05, 0.5)
    print(f"[INFO] Contamination adaptative : {contamination:.3f}")

    local_models = []
    for i, (Xc, _) in enumerate(clients_data):
        iso = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=i,
            n_jobs=-1
        )
        iso.fit(Xc)
        local_models.append(iso)
        print(f"  Client {i+1:02d} : entraîné ✓")

    # Agrégation : moyenne des scores
    all_scores = np.array([m.decision_function(X_test)
                           for m in local_models])
    avg_scores = all_scores.mean(axis=0)

    # Threshold calibré sur train (pas percentile fixe)
    train_scores = np.array([m.decision_function(X_train)
                             for m in local_models]).mean(axis=0)
    threshold = np.percentile(train_scores, contamination * 100)
    y_pred = (avg_scores < threshold).astype(int)

    print("\n[RAPPORT — Isolation Forest Fédéré]")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Anomalie"]))
    return y_pred, avg_scores


# ════════════════════════════════════════════
# 7. HYBRIDE : MLP + Isolation Forest
# ════════════════════════════════════════════

def hybrid_federated_ids(mlp_pred, iso_pred, y_test):
    print("\n" + "="*55)
    print("  SYSTÈME HYBRIDE — MLP Fédéré + IF Fédéré")
    print("="*55)

    # Logique OU : alerte si l'un des deux détecte
    hybrid = np.where((mlp_pred == 1) | (iso_pred == 1), 1, 0)

    acc = accuracy_score(y_test, hybrid)
    f1  = f1_score(y_test, hybrid, average='weighted', zero_division=0)
    print(f"\n  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score : {f1:.4f}")
    print(classification_report(y_test, hybrid,
                                target_names=["Normal", "Attaque"]))
    return hybrid


# ════════════════════════════════════════════
# 8. VISUALISATIONS
# ════════════════════════════════════════════

def plot_convergence(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Convergence du modèle fédéré", fontsize=13, fontweight='bold')

    ax1.plot(history['round'], history['accuracy'], 'b-o', label='Accuracy', linewidth=2)
    ax1.plot(history['round'], history['f1'],       'g-s', label='F1-Score',  linewidth=2)
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Seuil 90%')
    ax1.set_title("Accuracy & F1 par round")
    ax1.set_xlabel("Round de communication")
    ax1.set_ylabel("Score")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.05)

    ax2.plot(history['round'], history['loss'], 'r-^', linewidth=2, label='Loss moy. clients')
    ax2.set_title("Loss des clients par round")
    ax2.set_xlabel("Round de communication")
    ax2.set_ylabel("Binary Cross-Entropy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150, bbox_inches='tight')
    print("[INFO] Sauvegardé : convergence.png")
    plt.show()


def plot_confusion_matrices(y_test, mlp_pred, iso_pred, hybrid_pred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Matrices de confusion", fontsize=13, fontweight='bold')
    labels = ["Normal", "Attaque"]
    for pred, title, ax in [
        (mlp_pred,    "MLP Fédéré",     axes[0]),
        (iso_pred,    "IF Fédéré",      axes[1]),
        (hybrid_pred, "Hybride Fédéré", axes[2]),
    ]:
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average='weighted', zero_division=0)
        cm  = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{title}\nacc={acc:.3f} | f1={f1:.3f}")
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
    print("[INFO] Sauvegardé : confusion_matrices.png")
    plt.show()


def plot_comparison(y_test, mlp_pred, iso_pred, hybrid_pred):
    models = ['MLP\nFédéré', 'IF\nFédéré', 'Hybride\nFédéré']
    accs   = [accuracy_score(y_test, p) for p in [mlp_pred, iso_pred, hybrid_pred]]
    f1s    = [f1_score(y_test, p, average='weighted', zero_division=0)
              for p in [mlp_pred, iso_pred, hybrid_pred]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars_a = ax.bar(x - 0.2, accs, 0.35, label='Accuracy', color='steelblue')
    bars_f = ax.bar(x + 0.2, f1s,  0.35, label='F1-Score',  color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15)
    ax.set_title("Comparaison des modèles fédérés", fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha='center', fontsize=9)
    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
    print("[INFO] Sauvegardé : comparison.png")
    plt.show()


# ════════════════════════════════════════════
# 9. PIPELINE PRINCIPAL
# ════════════════════════════════════════════

def main():
    # ── Configuration ───────────────────────
    DATASET_PATH  = ".\\Network_complet.csv"
    N_CLIENTS     = 5
    N_ROUNDS      = 15
    LOCAL_EPOCHS  = 8
    LEARNING_RATE = 0.005
    IID           = True
    # ────────────────────────────────────────

    print("╔══════════════════════════════════════════════╗")
    print("║  IDS Fédéré TON_IoT — Version 2 (corrigée)  ║")
    print("╚══════════════════════════════════════════════╝\n")
    np.random.seed(42)

    # 1. Données
    X, y = load_and_preprocess(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Distribution clients
    clients_data = split_data_among_clients(X_train, y_train, N_CLIENTS, iid=IID)

    # 3. MLP Fédéré
    global_net, history, mlp_pred = federated_training(
        clients_data, X_test, y_test,
        n_rounds=N_ROUNDS, lr=LEARNING_RATE, local_epochs=LOCAL_EPOCHS
    )

    # 4. Isolation Forest Fédéré
    iso_pred, _ = federated_isolation_forest(
        clients_data, X_train, y_train, X_test, y_test
    )

    # 5. Hybride
    hybrid_pred = hybrid_federated_ids(mlp_pred, iso_pred, y_test)

    # 6. Visualisations
    plot_convergence(history)
    plot_confusion_matrices(y_test, mlp_pred, iso_pred, hybrid_pred)
    plot_comparison(y_test, mlp_pred, iso_pred, hybrid_pred)

    print("\n[DONE] Version 2 terminée.")


if __name__ == "__main__":
    main()
