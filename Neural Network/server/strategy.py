import os
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar

from .config import (
    MIN_AVAILABLE_CLIENTS, MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS,
    FRACTION_FIT, FRACTION_EVALUATE,
    LOCAL_EPOCHS, BATCH_SIZE, MODEL_DIR
)


os.makedirs(MODEL_DIR, exist_ok=True)

class SaveModelStrategy(FedAvg):
    """FedAvg strategy with custom behavior for Regression.
    - Saves the global model weights to `MODEL_DIR` after each fit round.
    - Aggregates evaluation metrics (MAE, MSE).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_fit_config_fn(self, server_round: int) -> Dict[str, Scalar]:
        """Return configuration (dict) sent to clients for local training."""
        return {
            "local_epochs": LOCAL_EPOCHS, 
            "batch_size": BATCH_SIZE,
            "round": server_round
        }

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Call base aggregation then persist the aggregated global weights."""
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            file_path = os.path.join(MODEL_DIR, "global_model.npz")
            np.savez(file_path, *weights)
            print(f"[Server] Saved global model at round {server_round} to {file_path}")

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results (Loss/MSE/MAE)."""
        
        if not results:
            return None, {}

        # Default aggregation of loss (MSE)
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Custom aggregation for MAE or other metrics
        total_examples = 0
        mae_weighted_sum = 0.0

        for client_proxy, eval_res in results:
            n = eval_res.num_examples
            total_examples += n
            
            # Clients return 'accuracy' or custom metrics. 
            # In our updated client, we don't return 'accuracy' but we might pass metrics in the future.
            # Currently, the loss (MSE) is the main indicator.
            
            # If you updated client to return {"mae": val}, extract it here:
            mae = eval_res.metrics.get("mae")
            if mae is not None:
                mae_weighted_sum += mae * n

        aggregated_mae = mae_weighted_sum / total_examples if total_examples > 0 else 0.0

        loss_str = f"{aggregated_loss:.4f}" if aggregated_loss is not None else "n/a"
        mae_str = f"{aggregated_mae:.4f}" if aggregated_mae > 0 else "n/a"
        
        print(f"[Server] Round {server_round} — Loss(MSE): {loss_str}, MAE: {mae_str}")

        return aggregated_loss, {"mae": aggregated_mae}

def get_strategy() -> SaveModelStrategy:
    """Factory returning the tuned strategy instance for the server."""
    return SaveModelStrategy(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    )
