import flwr as fl

# =========================
# STRATEGY
# =========================
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # 100% des clients participent à chaque round
    fraction_evaluate=1.0,
    min_fit_clients=1,          # minimum 1 client pour démarrer
    min_evaluate_clients=1,
    min_available_clients=1,    # attendre au moins 1 client connectés
)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("[INFO] Starting Flower Server on localhost:8090 ...")

    fl.server.start_server(
        server_address="localhost:8090",
        config=fl.server.ServerConfig(num_rounds=8),
        strategy=strategy,
    )

    print("[INFO] Federated Learning terminé.")
