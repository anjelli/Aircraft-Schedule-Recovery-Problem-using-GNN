import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGNN(nn.Module):
    def __init__(self, in_dims, hidden=64, num_layers=2, dropout=0.3):
        super().__init__()
        # Linear input encoders per node type
        self.input_lin = nn.ModuleDict({
            nt: nn.Linear(dim, hidden) for nt, dim in in_dims.items()
        })

        # Build heterogeneous conv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('aircraft', 'operates', 'flight'): SAGEConv(hidden, hidden),
                ('flight', 'next', 'flight'): SAGEConv(hidden, hidden),
                ('flight', 'at', 'airport'): SAGEConv(hidden, hidden),
                ('aircraft', 'connects', 'aircraft'): SAGEConv(hidden, hidden),
            }, aggr='mean')
            self.convs.append(conv)

        # Classifier on aircraft nodes
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        # initial linear transform for each node type
        x = {nt: F.relu(self.input_lin[nt](feat)) for nt, feat in x_dict.items()}

        for conv in self.convs:
            # filter edge_index_dict to keep only valid and non-empty relations
            filtered = {}
            for rel, eidx in edge_index_dict.items():
                if eidx is None or (isinstance(eidx, torch.Tensor) and eidx.numel() == 0):
                    continue
                src, _, dst = rel
                if (src not in x) or (dst not in x):
                    continue
                filtered[rel] = eidx

            if len(filtered) == 0:
                # skip if no valid edges
                continue

            x = conv(x, filtered)
            for k in x.keys():
                x[k] = F.relu(x[k])

        logits = self.classifier(x['aircraft']).squeeze(-1)
        return logits


# Optional test entry point (for standalone debugging)
if __name__ == "__main__":
    import pandas as pd
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from src.graph_builder import DayScheduleGraphBuilder
    from src.train import train

    flights = pd.read_csv("data/flights_canonical.csv")
    aircraft = pd.read_csv("data/aircraft_canonical.csv")
    disruptions = pd.read_csv("data/disruptions_synthetic.csv")

    builder = DayScheduleGraphBuilder(flights, aircraft)
    data_list, mappings = [], []
    for _, r in disruptions.iterrows():
        d = r.to_dict()
        data, mapping = builder.build_graph_for_disruption(d)
        y = torch.zeros(data["aircraft"].x.shape[0], dtype=torch.float)
        idx = mapping["tail2aircraft_idx"].get(d["disrupted_tail"], None)
        if idx is not None:
            y[idx] = 1.0
        data["aircraft"].y = y
        data_list.append(data)
        mappings.append(mapping)

    train_loader = PyGDataLoader(data_list, batch_size=8, shuffle=True)
    val_loader = None
    in_dims = {
        "aircraft": data_list[0]["aircraft"].x.shape[1],
        "flight": data_list[0]["flight"].x.shape[1],
        "airport": data_list[0]["airport"].x.shape[1],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroGNN(in_dims, hidden=128, num_layers=2, dropout=0.3).to(device)
    model = train(model, train_loader, val_loader, device, epochs=10, lr=1e-3)
    torch.save({"model_state": model.state_dict(), "in_dims": in_dims},
               "data/gnn_model_dataset.pth")
    print("Model training complete. Saved to data/gnn_model_dataset.pth")
