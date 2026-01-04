import flwr as fl
import numpy as np
import os
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import List, Tuple
from flwr.common import Metrics

# Define the checkpoint file
CHECKPOINT_FILE = "global_model.npz"

# Calculate Weighted Average for Metrics
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    
    # We look for "mae" because your client sends {"mae": ...}
    # If your client sends "loss", change this to "loss"
    accuracies = [num_examples * m.get("mae", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if sum(examples) == 0:
        return {"mae": 0}

    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(accuracies) / sum(examples)}

# Load Model
initial_parameters = None
if os.path.exists(CHECKPOINT_FILE):
    print(f"[Server] Loading existing global model from {CHECKPOINT_FILE}...")
    try:
        data = np.load(CHECKPOINT_FILE)
        loaded_weights = [data[key] for key in data.files]
        initial_parameters = ndarrays_to_parameters(loaded_weights)
    except Exception as e:
        print(f" Failed to load checkpoint: {e}. Starting fresh.")
else:
    print(" No checkpoint found. Starting from scratch.")

# Strategy
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call the standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Save the result to a file
        if aggregated_parameters is not None:
            print(f" Saving round {server_round} global model to {CHECKPOINT_FILE}...")
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            np.savez(CHECKPOINT_FILE, *aggregated_ndarrays)
                
        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(
    fraction_fit=1.0,           
    fraction_evaluate=1.0,
    min_fit_clients=5,           
    min_available_clients=5,     
    initial_parameters=initial_parameters, 
    fit_metrics_aggregation_fn=weighted_average, 
    evaluate_metrics_aggregation_fn=weighted_average,
)


from config import NUM_ROUNDS 

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
