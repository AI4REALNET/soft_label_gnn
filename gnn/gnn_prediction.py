"""
Similar to the junior, but replaces the neural network by a graph neural network to embed the gruid
"""
import logging
import os.path
import pickle
import time
from collections import ChainMap
from pathlib import Path
from typing import Optional, TypedDict, Union

import grid2op
import numpy as np
import torch
import torch_geometric.data
from lightsim2grid import LightSimBackend
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LinearLR

from gnn.gnn_models import GAT
from gnn.torch_geometric_datasets import Grid2opGraphDataset, Grid2opGraphDatasetProcessed
import optuna
import  matplotlib.pyplot as plt



class GnnPrediction():
    def __init__(self, config):
        """

        Args:
            config: parameters of the gnn model
        """
        self.config = config
        self.num_output = config.get("num_output", 2030)
        self.model = self._build_model(config)
        self.loss_fn = config.get("loss", torch.nn.NLLLoss())
        self.algorithm_config = config.get("kwargs_algorithm", {"lr": 0.01, "weight_decay": 5e-4})
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _build_model(self, config: Optional[dict] = None) -> torch.nn.Module:
        """

        Args:
            config: parameters of the gnn model

        Returns:

        """
        if config != None:
            config["out_channels_lin"]=self.num_output
            model = GAT(config)

            
        else:
            self.default_config = {
                "in_channels": 27,
                "out_channels_conv": 128,
                "out_channels_lin": self.num_output,
            }
            model = GAT()
        return model

    def train(
            self,
            dataset_train: torch_geometric.data.Dataset,
            dataset_val: torch_geometric.data.Dataset,
            save_dir: Path,
            epochs=1,
            save_every_x_steps: int = 5,
            num_workers: Optional[int] = None,
            batchsize: int = 64,
            trial = None

    ):
        """Train the GNN junior model for given number of epochs.

        Args:
            dataset_train: Train dataset containing grid observations as graph including node features, adjacency matrix and target
            dataset_val: Validation dataset containing grid observations as graph including node features,
            adjacency matrix and target
            save_dir: Director, where to save the model
            epochs: Number of epochs for the training
            save_every_x_steps: How often should the model be saved in the training process. If none, the model is
            only saved at the end
            batchsize: Size of batches

        Returns: Returns the loss report and val loss report.

        """
        if not save_dir.is_dir():
            logging.warning(f"{save_dir} does not exists yet. Create directory")
            save_dir.mkdir(parents=True, exist_ok=True)
        logging.warning(f"The model {self.model.__class__.__name__} will trained for {epochs} epochs on {self.device} with config {self.algorithm_config} . The checkpoints will be saved to {save_dir} every {save_every_x_steps} steps.")
        print(f"The model {self.model.__class__.__name__} will trained for {epochs} epochs on {self.device} with config {self.algorithm_config} . The checkpoints will be saved to {save_dir} every {save_every_x_steps} steps.")

        with open(save_dir / 'train_config.pkl', 'wb') as f:
            pickle.dump([self.config], f)
        model = self.model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), **self.algorithm_config)
        #if self.config["lr_scheduler"]:
            #scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=1000)

        if self.config.get("lr_scheduler"):
            scheduler_config = self.config["lr_scheduler"]
            scheduler_type = scheduler_config["type"]
            scheduler_params = scheduler_config["params"]
            
            if scheduler_type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
            elif scheduler_type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
            elif scheduler_type == "LinearLR":
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")



        if num_workers is None:
            num_workers = 0
        if self.device == "cuda" and num_workers > 0:
            logging.warning(f"The device is set to {self.device} and num_workers is {num_workers}. This will most "
                            f"likely into errors due to pickle problems of Grid2op. We advice to either set workers "
                            f"to 0 or only use CPU")

        dl_train = DataLoader(dataset_train, batch_size=batchsize, num_workers=num_workers, shuffle=True, drop_last=True)

        loss_report = []
        val_loss_report = []
        val_acc_report = []
        best_val_loss = 20000.0
        for i in range(epochs):
            iteration_loss_train = []
            model.train()
            for s_train in dl_train:
                s_train = s_train.to(self.device)
                optimizer.zero_grad()
                pred = model(s_train)
                if self.config["hard_label"]:
                    loss = self.loss_fn(pred, s_train.y)
                else:
                    loss = self.loss_fn(pred, s_train.y.reshape(batchsize,-1))
                iteration_loss_train.append(loss.item())
                loss.backward()
                optimizer.step()
            train_loss = np.mean(iteration_loss_train)
            loss_report.append(train_loss)

            model.eval()
            val_loss, val_acc = self.validate(dataset=dataset_val, batchsize=batchsize)
            val_loss_report.append(val_loss)
            val_acc_report.append(val_acc)
            
            
            before_lr = optimizer.param_groups[0]["lr"]

            if self.config.get("lr_scheduler"):
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(val_loss)  # Adjust learning rate based on validation loss
                else:
                    scheduler.step()  # For other schedulers
            after_lr = optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.4f -> %.4f" % (i, before_lr, after_lr))
 
            # Save model every x steps:
            if not save_every_x_steps is None:
                if i % save_every_x_steps == 0:
                    torch.save(model.state_dict(), save_dir / "model.pt")
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_dir / "model.pt")

            t = time.strftime("%H:%M:%S", time.localtime())
            logging.warning(f"{t}: Episode {i} completed with training loss {train_loss} and validation loss {val_loss} and validation accuracy {val_acc}")
            with open(save_dir / 'loss_report.pkl', 'wb') as f:
                pickle.dump([loss_report, val_loss_report, val_acc_report], f)
            torch.cuda.empty_cache()
            # Report the validation loss to Optuna
        if trial:
            trial.report(val_loss, i)
            # Prune if needed
            if trial.should_prune():
                raise optuna.TrialPruned()

        torch.save(model.state_dict(), save_dir / "model.pt")

        return loss_report, val_loss_report

    @torch.no_grad()
    def validate(self, dataset, batchsize: Optional[int] = 1):
        """

        Args:
            dataset (): Validation dataset containing grid observations as graph including node features, adjacency matrix and target
        Returns:

        """
        model = self.model.to(self.device)
        model.eval()
        dl = DataLoader(dataset, batch_size=batchsize, num_workers=0, drop_last=True)
        iteration_loss_val = []
        correct = 0
        total = 0
        for val_d in dl:
            val_d = val_d.to(self.device)
            pred = model(val_d)
            #pred_class = pred.argmax(dim=1)
            #correct += (pred_class == val_d.y.reshape(batchsize,-1)).sum().item()
            #total += val_d.y.size(0)
            if self.config["hard_label"]:
                loss = self.loss_fn(pred, val_d.y)
            else:
                loss = self.loss_fn(pred, val_d.y.reshape(batchsize,-1))
            iteration_loss_val.append(loss.item())
        #acc = correct / total if total > 0 else 0
        return np.mean(iteration_loss_val), 0#acc

    def test(self, dataset, save_path: Optional[Union[str, Path]] = None) -> dict:

        if isinstance(save_path, Path):
            self.model.load_state_dict(torch.load(save_path, map_location=torch.device(self.device.type)))
            logging.info(f"Imported model from{save_path}")
        elif isinstance(save_path, str):
            self.model.load_state_dict(torch.load(Path(save_path, map_location=torch.device(self.device.type))))
            logging.info(f"Imported model from{save_path}")
            
            
            
        model = self.model.to(self.device)
        dl = DataLoader(dataset, batch_size=1000, num_workers=0, shuffle=False)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dl:
                data = data.to(self.device)
                out = model(data)

                # Assuming `out` is logits and `data.y` are class indices
                preds = out.argmax(dim=1)
                correct += (preds == data.y).sum().item()
                total += data.y.size(0)

        return correct / total if total > 0 else 0

def train(
        run_name: str,
        dataset_path: Path,
        dataset_name_train: str = "junior_dataset_train",
        dataset_name_val: str = "junior_dataset_val",
        target_model_path: Path = Path("model_ckpt"),
        epochs: int =20,
        env: str ="l2rpn_case14_sandbox",
        cfg: dict =None):

    if not os.path.isdir(target_model_path):
        logging.warning(f"{target_model_path} does not exists yet. Create directory")
        os.mkdir(f"{target_model_path}")

    # logging.basicConfig( encoding="utf-8", level=logging.WARN)
    logging.basicConfig(
        filename=f"{target_model_path}/experiments.log", encoding="utf-8", level=logging.WARN
    )
    backend = LightSimBackend()
    env = grid2op.make(env, backend=backend)
    d_train = Grid2opGraphDataset(root=dataset_path, dataset_name=dataset_name_train, env=env, split="train",
                                  include_disconnected_lines=True)

    d_val = Grid2opGraphDataset(root=dataset_path, dataset_name=dataset_name_val, env=env, split="val",
                                include_disconnected_lines=True)
    if cfg is None:
        cfg = {
        "in_channels": 11,
        "out_channels_conv": 256,
        "out_channels_lin": np.unique((d_train.a)).shape[0],
        "kwargs_algorithm": {"lr": 0.001,
        "weight_decay": 5e-4},
        "dropout": 0.5,
        "lr_scheduler": True
        }

    predictor = GnnPrediction(config=cfg)

    loss_report, val_loss_report= predictor.train(d_train, d_val, save_dir=target_model_path, epochs=epochs, save_every_x_steps=None, batchsize=256)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss_report)), loss_report, linewidth=2.0, color = "r")
    ax.plot(np.arange(len(val_loss_report)), val_loss_report, linewidth=2.0, color = "b")
    ax.legend(['train', 'val'])
    plt.savefig(f"loss_plot_{run_name}", format="svg")
    return loss_report, val_loss_report

def train_fast(
        run_name: str,
        dataset_path: Path,
        dataset_name_train: str = "junior_dataset_train",
        dataset_name_val: str = "junior_dataset_val",
        target_model_path: Path = Path("model_ckpt"),
        epochs: int =20,
        env: str ="l2rpn_case14_sandbox",
        cfg: dict =None,
        global_features: bool=False,
        trial = None):

    if not os.path.isdir(target_model_path):
        logging.warning(f"{target_model_path} does not exists yet. Create directory")
        os.mkdir(f"{target_model_path}")

    # logging.basicConfig( encoding="utf-8", level=logging.WARN)
    logging.basicConfig(
        filename=f"{target_model_path}/experiments.log", encoding="utf-8", level=logging.WARN
    )
    backend = LightSimBackend()
    env = grid2op.make(env, backend=backend)

    if cfg is None:
        cfg = {
            "lr_scheduler": True,
            "kwargs_algorithm":{"lr": 0.001,"weight_decay": 5e-4},
            "in_channels": 27,
            "gat_layers": [
                {"out_channels": 16, "heads": 4, "dropout": 0.2, "concat": True},
                {"out_channels": 256, "heads": 1, "dropout": 0.2, "concat": False}
            ],
            "linear_layers": [
                {"out_features": 512, "dropout": 0.0},
                {"out_features": 1024, "dropout": 0.0},
                {"out_features": 4} # change according to output
            ],
            "dropout": 0.0,
            "pooling_type": "max",
            "gat_activation": "elu",  # Activation function for GAT layers
            "linear_activation": "relu",  # Activation function for linear layers
            "batch_size": 256
        }

    d_train = Grid2opGraphDatasetProcessed(root=dataset_path, dataset_name=dataset_name_train, env=env,
                                           split="train",
                                           include_disconnected_lines=True)

    d_val = Grid2opGraphDatasetProcessed(root=dataset_path, dataset_name=dataset_name_val, env=env, split="val",
                                         include_disconnected_lines=True)
    if global_features:
        cfg["global_features_size"] = d_train.get(0).global_features.shape[0]
    else:
        cfg["global_features_size"] = 0

    


    predictor = GnnPrediction(config=cfg)

    loss_report, val_loss_report= predictor.train(d_train, d_val, save_dir=target_model_path, epochs=epochs, save_every_x_steps=None, batchsize=cfg["batch_size"], trial=trial)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss_report)), loss_report, linewidth=2.0, color = "r")
    ax.plot(np.arange(len(val_loss_report)), val_loss_report, linewidth=2.0, color = "b")
    ax.legend(['train', 'val'])
    plt.savefig(f"loss_plot_{run_name}", format="svg")
    return loss_report, val_loss_report

