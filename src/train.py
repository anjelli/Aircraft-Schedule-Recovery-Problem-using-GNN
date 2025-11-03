import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
import numpy as np

# ---------------------------------------------------------
# Utility: validate and fix hetero graphs before training
# ---------------------------------------------------------
def validate_and_fix_hetero(data):
    issues = []
    # 1) Ensure each node type has .x
    for ntype in data.node_types:
        if 'x' not in data[ntype] or data[ntype].x is None:
            n_nodes = len(data[ntype].node_ids) if hasattr(data[ntype], 'node_ids') else 0
            data[ntype].x = torch.zeros((n_nodes, 1), dtype=torch.float)
            issues.append(f"Missing .x for node type '{ntype}' -> created zeros [{n_nodes},1]")
        elif not torch.is_tensor(data[ntype].x):
            data[ntype].x = torch.tensor(data[ntype].x, dtype=torch.float)
    # 2) Validate edge indices
    for edge_type in data.edge_types:
        try:
            eidx = data[edge_type].edge_index
        except Exception:
            eidx = None
        if eidx is None:
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            issues.append(f"{edge_type}: edge_index missing -> empty created")
            continue
        if not torch.is_tensor(eidx):
            data[edge_type].edge_index = torch.tensor(eidx, dtype=torch.long)
        if data[edge_type].edge_index.ndim != 2 or data[edge_type].edge_index.size(0) != 2:
            issues.append(f"{edge_type}: bad shape {data[edge_type].edge_index.shape}")
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            continue
        src_type, _, dst_type = edge_type
        n_src = data[src_type].x.shape[0]
        n_dst = data[dst_type].x.shape[0]
        src_idx = data[edge_type].edge_index[0]
        dst_idx = data[edge_type].edge_index[1]
        if src_idx.numel() > 0:
            mask = (src_idx < n_src) & (dst_idx < n_dst)
            if mask.sum() < mask.numel():
                data[edge_type].edge_index = data[edge_type].edge_index[:, mask]
                issues.append(f"{edge_type}: removed {mask.numel()-mask.sum().item()} invalid edges")
    return issues


# ---------------------------------------------------------
# Metric and evaluation
# ---------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.3):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            validate_and_fix_hetero(data)
            logits = model(data.x_dict, data.edge_index_dict)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(data['aircraft'].y.cpu().numpy())
            ps.append(prob)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return compute_metrics(y, p), y, p


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    pos_weight = torch.tensor([float((y==0).sum() / max((y==1).sum(),1))])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    best, best_rec = None, -1
    for ep in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            validate_and_fix_hetero(data)
            opt.zero_grad()
            logits = model(data.x_dict, data.edge_index_dict)
            loss = loss_fn(logits, data['aircraft'].y.to(device))
            loss.backward()
            opt.step()
        print(f"Epoch {ep}: loss={loss.item():.4f}")
        if val_loader is not None:
            metrics, _, _ = evaluate(model, val_loader, device)
            if metrics['recall'] > best_rec:
                best_rec = metrics['recall']
                best = {k: v.cpu() for k, v in model.state_dict().items()}
    if best:
        model.load_state_dict(best)
    return model


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from src.graph_builder import DayScheduleGraphBuilder
    from src.model import HeteroGNN

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
        validate_and_fix_hetero(data)
        data_list.append(data)
        mappings.append(mapping)

    # --- loaders ---
    train_loader = PyGDataLoader(data_list, batch_size=8, shuffle=True)
    val_loader = None

    # --- debug first graph ---
    first = train_loader.dataset[0]
    print("Node feature shapes:")
    for nt in first.node_types:
        print(f"  {nt}: {first[nt].x.shape}")
    print("Edge types:", list(first.edge_types))

    in_dims = {
        "aircraft": first["aircraft"].x.shape[1],
        "flight": first["flight"].x.shape[1],
        "airport": first["airport"].x.shape[1],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroGNN(in_dims, hidden=128, num_layers=2, dropout=0.3).to(device)
    model = train(model, train_loader, val_loader, device, epochs=10, lr=1e-3)

    torch.save({"model_state": model.state_dict(), "in_dims": in_dims},
               "data/gnn_model_dataset.pth")
    print("Training complete. Model saved to data/gnn_model_dataset.pth")
