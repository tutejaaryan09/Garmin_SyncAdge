"""Server configuration constants.

Keep training / federation-related constants here. These variables are
imported by `server/server_flwr.py` and `server/strategy.py`.
"""

# Number of federated learning rounds to run
NUM_ROUNDS = 50  # Enough rounds for everyone to be picked a few times

# Address the Flower server listens on (host:port)
SERVER_ADDRESS = "localhost:8080"

# Directory where the server persists the global model
MODEL_DIR = "checkpoints"

# --- AUTO APPROACH CONFIGURATION ---

# 1. Wait for ALL 44 clients to connect before starting the first round
# (This prevents the round starting with just the fast clients)
MIN_AVAILABLE_CLIENTS = 5

# 2. Select ONLY 5 clients per round for training
MIN_FIT_CLIENTS = 5
MIN_EVALUATE_CLIENTS = 5

# 3. Fraction to select: 5 / 44 ≈ 0.113
# We set it slightly higher (0.12) to be safe, or just rely on MIN_FIT_CLIENTS logic.
FRACTION_FIT = 1.0    # Selects roughly 5-6 clients
FRACTION_EVALUATE = 1.0 # Evaluate on 5-6 clients (faster) 
                         # OR set to 1.0 if you want to test on everyone every round (slower)

# Local training defaults sent to clients
LOCAL_EPOCHS = 10  # A balance between speed and learning
BATCH_SIZE = 16

