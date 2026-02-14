import warnings
from gnn.gnn_prediction import train, train_fast, GnnPrediction
from pathlib import Path
warnings.filterwarnings("ignore")
import optuna
import random
import torch.nn as nn
def objective(trial):
    # Dataset paths
    dataset_path = "./data/norm"
    dataset_name_train = "soft_target_data_train_processed_norm.pt"
    dataset_name_val = "soft_target_data_val_processed_norm.pt"
    
    # Hyperparameter space
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    
    # GAT layers
    num_gat_layers = trial.suggest_int("num_gat_layers", 2, 6)  # Tune number of GAT layers
    gat_layers = []
    for i in range(num_gat_layers):
        out_channels = trial.suggest_categorical(f"gat_out_channels_{i}", [16, 32, 64, 128])
        heads = trial.suggest_categorical(f"gat_heads_{i}", [1, 2, 4, 8])
        dropout = trial.suggest_uniform(f"gat_dropout_{i}", 0.0, 0.5)
        concat = i < (num_gat_layers - 1)  # Always concat except for the last layer
        gat_layers.append({
            "out_channels": out_channels,
            "heads": heads,
            "dropout": dropout,
            "concat": concat
        })

    pooling_type = trial.suggest_categorical("pooling_type", ["max", "mean", "add"])
    
    # Linear layers
    num_linear_layers = trial.suggest_int("num_linear_layers", 2, 4)  # Tune number of linear layers
    linear_layers = []
    for i in range(num_linear_layers):
        out_features = trial.suggest_categorical(f"linear_out_features_{i}", [128, 256, 512, 1024])
        dropout = trial.suggest_uniform(f"linear_dropout_{i}", 0.0, 0.5)
        linear_layers.append({"out_features": out_features, "dropout": dropout})
    # Add final output layer
    linear_layers.append({"out_features": 2030})  # Adjust output size if needed
    
    # Scheduler
    scheduler_params = {
        "mode": "min",
        "factor": 0.9,
        "patience": 10,
        "verbose": True, 
        "min_lr": 1e-9
    }
    
    # Model configuration
    cfg = {
        "lr_scheduler": {"type": "ReduceLROnPlateau", "params": scheduler_params},
        "kwargs_algorithm": {"lr": learning_rate, "weight_decay": weight_decay},
        "in_channels": 27,
        "gat_layers": gat_layers,
        "linear_layers": linear_layers,
        "dropout": 0.0, #trial.suggest_uniform("global_dropout", 0.0, 0.5),
        "pooling_type": pooling_type,
        "gat_activation": "elu",
        "linear_activation": "relu",
        "batch_size": 256, #trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        "loss": nn.KLDivLoss(reduction='batchmean'),
        "hard_label": False

    }
    
    # Train the model
    loss_report, val_loss_report = train_fast(
        run_name=f"optuna_trial_{trial.number}",
        dataset_path=dataset_path,
        dataset_name_train=dataset_name_train,
        dataset_name_val=dataset_name_val,
        target_model_path=Path(f"./saved_models/optuna_distributed_pruned/optuna_trial_{trial.number + random.randint(1, 100000)}"),
        epochs=1000,
        env="l2rpn_wcci_2022",
        cfg=cfg,
        trial=trial
    )
    
    # Use validation loss for objective
    return min(val_loss_report)

if __name__ == "__main__":
    # Create Optuna study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=50)

    study = optuna.create_study(
        study_name="distributed_optimization",
        direction="minimize",  # Or "maximize", depending on your goal
        storage="sqlite:///saved_models/optuna_distributed_pruned/study.db",  # Using SQLite for file storage
        load_if_exists=True,
        pruner=pruner

    )
    
    study.optimize(objective, n_trials=100)
    
    # Print best trial
    print("Best trial:")
    print(study.best_trial)




