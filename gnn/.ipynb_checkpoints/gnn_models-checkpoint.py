import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, Dropout, functional as F
from torch_geometric.graphgym import GCNConv
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

class GAT(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.default_config = {
            "in_channels": 27,
            "gat_layers": [
                {"out_channels": 16, "heads": 4, "dropout": 0.2, "concat": True},
                {"out_channels": 256, "heads": 1, "dropout": 0.2, "concat": False}
            ],
            "linear_layers": [
                {"out_features": 512, "dropout": 0.3},
                {"out_features": 1024, "dropout": 0.3},
                {"out_features": 4} # change according to output
            ],
            "dropout": 0.0,
            "pooling_type": "max",
            "gat_activation": "elu",  # Activation function for GAT layers
            "linear_activation": "relu"  # Activation function for linear layers
        }

        # Use provided config or fall back to default
        if config is None:
            config = self.default_config
        self.config = config

        self.global_features_size = config.get("global_features_size", 0)
        self.loss = config.get("loss", nn.CrossEntropyLoss())

        # Set up GAT layers
        self.gat_layers = torch.nn.ModuleList()
        in_channels = config["in_channels"]
        for layer_cfg in config["gat_layers"]:
            out_channels = layer_cfg["out_channels"]
            heads = layer_cfg["heads"]
            concat = layer_cfg.get("concat", True)
            dropout = layer_cfg.get("dropout", config["dropout"])
            self.gat_layers.append(
                GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
            )
            in_channels = out_channels * heads if concat else out_channels

        # Set up linear layers
        self.linear_layers = torch.nn.ModuleList()
        self.linear_dropouts = []

        # add to linear layer in_channels the number of global_features
        in_channels = in_channels + self.global_features_size
        for layer_cfg in config["linear_layers"]:
            out_features = layer_cfg["out_features"]
            dropout = layer_cfg.get("dropout", config["dropout"])
            layer = Linear(in_channels, out_features)
            self.linear_layers.append(layer)
            self.linear_dropouts.append(dropout)  # Register dropout layers

            in_channels = out_features

        # Define the pooling and activation functions based on config
        self.pooling_type = config["pooling_type"]
        self.gat_activation_func = getattr(F, config["gat_activation"], F.elu)
        self.linear_activation_func = getattr(F, config["linear_activation"], F.relu)

    def forward(self, data):
        global_features = None
        if self.global_features_size == 0:
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index, global_features = data.x, data.edge_index, data.global_features.float().reshape(-1,self.global_features_size)

        # Apply GAT layers with specified activation
        for layer in self.gat_layers:
            x = self.gat_activation_func(layer(x, edge_index))

        # Apply pooling
        if self.pooling_type == "max":
            x = global_max_pool(x, data.batch)
        elif self.pooling_type == "mean":
            x = global_mean_pool(x, data.batch)
        elif self.pooling_type == "add":
            x = global_add_pool(x, data.batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        if self.global_features_size > 0:
            x = torch.cat([x, global_features], dim=1)

       
        for i in range(len(self.linear_layers)):

            if i == len(self.linear_layers) - 1:
                x = self.linear_layers[i](x)
            else:
                x = self.linear_activation_func(self.linear_layers[i](x))
                x = F.dropout(x, p=self.linear_dropouts[i], training=self.training)

        
        if isinstance(self.loss, nn.CrossEntropyLoss):
            return x
        else:
            return F.log_softmax(x, dim=1)
