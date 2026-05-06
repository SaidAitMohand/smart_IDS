import pandas as pd
import os

# ─────────────────────────────────────────────
# Colonnes du dataset NSL-KDD (41 features + label + difficulty)
# ─────────────────────────────────────────────
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
    "label",        # ex: normal, neptune, smurf, ...
    "difficulty"    # score de difficulté (colonne présente dans NSL-KDD)
]

# ─────────────────────────────────────────────
# Paramètres — modifiez ces chemins si besoin
# ─────────────────────────────────────────────
INPUT_FILES = [
    "C:\\Users\\User\\Desktop\\pfe\\datasets\\NSL-KDD\\KDDTrain+.txt",   # fichier d'entraînement NSL-KDD
    "C:\\Users\\User\\Desktop\\pfe\\datasets\\NSL-KDD\\KDDTest+.txt",    # fichier de test  NSL-KDD  (optionnel)
]

OUTPUT_NORMAL  = "nslkdd_normal.csv"
OUTPUT_ATTACKS = "nslkdd_attacks.csv"

# ─────────────────────────────────────────────
# Lecture & concaténation des fichiers source
# ─────────────────────────────────────────────
frames = []
for path in INPUT_FILES:
    if os.path.exists(path):
        df = pd.read_csv(path, header=None, names=COLUMNS)
        frames.append(df)
        print(f"[✓] Chargé : {path}  ({len(df):,} lignes)")
    else:
        print(f"[!] Fichier introuvable, ignoré : {path}")

if not frames:
    raise FileNotFoundError(
        "Aucun fichier NSL-KDD trouvé. "
        "Placez KDDTrain+.txt (et/ou KDDTest+.txt) dans le même dossier que ce script."
    )

data = pd.concat(frames, ignore_index=True)
print(f"\n[i] Total lignes chargées : {len(data):,}")

# ─────────────────────────────────────────────
# Séparation normal / attaques
# ─────────────────────────────────────────────
normal_df  = data[data["label"] == "normal"].copy()
attacks_df = data[data["label"] != "normal"].copy()

# ─────────────────────────────────────────────
# Statistiques
# ─────────────────────────────────────────────
print(f"\n{'─'*45}")
print(f"  Trafic normal  : {len(normal_df):>8,} lignes")
print(f"  Attaques       : {len(attacks_df):>8,} lignes")
print(f"{'─'*45}")

print("\nTypes d'attaques présentes :")
attack_counts = attacks_df["label"].value_counts()
for attack, count in attack_counts.items():
    print(f"   {attack:<25} {count:>7,}")

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────
normal_df.to_csv(OUTPUT_NORMAL, index=False)
attacks_df.to_csv(OUTPUT_ATTACKS, index=False)

print(f"\n[✓] Fichier normal   → {OUTPUT_NORMAL}")
print(f"[✓] Fichier attaques → {OUTPUT_ATTACKS}")
print("\nTerminé !")