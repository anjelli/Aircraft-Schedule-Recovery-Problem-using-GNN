import torch
from src.model import HeteroGNN
from src.graph_builder import DayScheduleGraphBuilder
import pandas as pd

def infer_one(builder, disruptions_df, model_path, device, threshold=0.5):
    ckpt = torch.load(model_path, map_location=device)
    in_dims = ckpt['in_dims']
    model = HeteroGNN(in_dims, hidden=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    d = disruptions_df.iloc[0].to_dict()
    data, mapping = builder.build_graph_for_disruption(d)
    data = data.to(device)

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        probs = torch.sigmoid(logits).cpu().numpy()

    tails = [mapping['aircraft_idx2tail'][i] for i in range(len(probs))]
    selected = [t for i, t in enumerate(tails) if probs[i] >= threshold]
    return selected, dict(zip(tails, probs.tolist()))

if __name__ == "__main__":
    flights = pd.read_csv("data/flights_canonical.csv")
    aircraft = pd.read_csv("data/aircraft_canonical.csv")
    disruptions = pd.read_csv("data/disruptions_synthetic.csv")

    builder = DayScheduleGraphBuilder(flights, aircraft)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected, probs = infer_one(builder, disruptions, "data/gnn_model_dataset.pth", device, threshold=0.05)

    print("All aircraft probabilities:")
    for t, p in probs.items():
        print(f"{t}: {p:.4f}")
    print("Predicted key aircraft:", selected)
