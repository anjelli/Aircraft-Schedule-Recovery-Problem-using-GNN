# Aircraft-Schedule-Recovery-Problem-using-GNN

# Disruption-Aware Aircraft Reassignment using Heterogeneous Graph Neural Networks and MILP Optimization

This repository contains the implementation for a two-phase framework combining **graph-based disruption prediction** with **optimization-based flight reassignment**.  
The work models airline operations as a dynamic heterogeneous network to identify *key aircraft* affected by potential disruptions and perform *targeted schedule recovery* using mixed-integer linear programming (MILP).

---

## Overview

Airline networks are tightly coupled: a disruption to one flight or aircraft can propagate across the schedule.  
This project integrates machine learning and optimization to mitigate such cascading effects.

**Phase 1 — Disruption Identification (GNN)**
- Constructs a heterogeneous flight–aircraft–airport graph per operational day.
- Uses a GraphSAGE-based **HeteroGNN** to learn relational dependencies.
- Identifies key aircraft most likely to be affected by disruptions.

**Phase 2 — Reassignment Optimization (MILP)**
- Selectively reassigns flights **only for the affected aircraft** predicted by the GNN.
- Maintains constraints on aircraft availability, turnaround time, and schedule continuity.
- Minimizes reassignment cost inversely proportional to aircraft reliability probability.

---

## 3. Repository Structuredata/

### data/

├── flights_canonical.csv # Flight schedule data

├── aircraft_canonical.csv # Aircraft metadata

├── disruptions_synthetic.csv # Disruption scenarios

├── gnn_model_dataset.pth # Trained HeteroGNN model checkpoint

### src/

├── graph_builder.py # Heterogeneous graph construction

├── model.py # HeteroGNN architecture (PyTorch Geometric)

├── train.py # GNN training and validation

├── inference.py # Key aircraft inference script

├── cplex_optimizer.py # MILP model for flight reassignment

├── visualize_results.py # (optional) Visualization utilities

└── utils/ # Supporting functions

---

## 4. Methodology Summary

| Component | Description |
|------------|-------------|
| **Graph Representation** | Captures spatial-temporal and operational dependencies among flights, aircraft, and airports. |
| **GNN Model** | Multi-layer HeteroConv (GraphSAGE) with ReLU and dropout regularization. |
| **Loss Function** | Binary Cross-Entropy with Logits (per-aircraft disruption label). |
| **Inference Output** | Aircraft disruption probabilities used to select top key aircraft. |
| **Optimizer** | PuLP-based MILP minimizing reassignment cost subject to operational constraints. |

---

## 5. Output

### **Training and Inference**
Node feature shapes:
  aircraft: torch.Size([48, 2])
  flight: torch.Size([171, 2])
  airport: torch.Size([47, 2])
Edge types: [('aircraft', 'operates', 'flight'), ('flight', 'connects', 'airport'), ('flight', 'operated_by', 'aircraft'), ('aircraft', 'connects', 'aircraft')]    
Epoch 1: loss=1.0247
Epoch 2: loss=0.4511
Epoch 3: loss=0.1269
Epoch 4: loss=0.1573
Epoch 5: loss=0.1098
Epoch 6: loss=0.1219
Epoch 7: loss=0.1015
Epoch 8: loss=0.1231
Epoch 9: loss=0.1129
Epoch 10: loss=0.1131
Training complete. Model saved to data/gnn_model_dataset.pth


### **Disruption Probabilities**
VTAIA: 0.0292
VTAIB: 0.0140
VTAIC: 0.0168
VTAIF: 0.0441
VTAIG: 0.0184
VTAIH: 0.0649
VTAII: 0.0286
VTAIO: 0.0122
VTAIP: 0.0461
VTAIR: 0.0235
VTBBD: 0.0156
VTBBE: 0.0157
VTBBJ: 0.0169
VTBBK: 0.0077
VTBBL: 0.0704
VTBBM: 0.0037
VTBBO: 0.0252
VTBBP: 0.0798
VTBBQ: 0.0343
VTBBT: 0.0167
VTICB: 0.0268
VTICD: 0.0214
VTICF: 0.0178
VTICG: 0.0586
VTICH: 0.0312
VTICN: 0.0236
VTICO: 0.0481
VTICQ: 0.0177
VTICX: 0.0096
VTXEA: 0.0167
VTXEB: 0.0165
VTXEC: 0.0247
VTXEF: 0.0181
VTXEG: 0.0066
VTXEH: 0.0275
VTXEI: 0.0149
VTXEJ: 0.0135
VTXEK: 0.0304
VTXEM: 0.0135
VTXEN: 0.0053
VTXEO: 0.0116
VTXEP: 0.0141
VTXEQ: 0.0258
VTXER: 0.0146
VTXES: 0.0206
VTXET: 0.0163
VTXEU: 0.0248
VTXEX: 0.0073

Predicted key aircraft: ['VTAIH', 'VTBBL', 'VTBBP', 'VTICG']


### **Optimization and Reassignment Results**

Total flights: 171

Affected flights (to reassign): 8

Fixed flights (kept as-is): 163

Affected counts per key tail: {'VTAIH': 2, 'VTBBL': 2, 'VTBBP': 2, 'VTICG': 2}

Schedule: [Google Sheet](https://docs.google.com/spreadsheets/d/1yMD7l4oskI0d8e_u2W4zq-zSbvu1V5dJ4vAMUYheTsE/edit?gid=110869346#gid=110869346)

Solver status: Optimal

Affected flights reassigned:
ZZ0640 -> VTBBL
ZZ0639 -> VTAIH
ZZ0627 -> VTAIH
ZZ0628 -> VTBBL
ZZ0549 -> VTBBP
ZZ0550 -> VTICG
ZZ0516 -> VTBBP
ZZ0515 -> VTBBP

---

## 6. Summary Statistics

| Assigned Tail | Total Flights | Reassigned | Avg STD | Avg STA |
|----------------|----------------|-------------|-----------|-----------|
| VTAIH | 2 | 2 | 502.5 | 607.5 |
| VTBBL | 2 | 2 | 477.5 | 577.5 |
| VTBBP | 3 | 3 | 478.3 | 570.0 |
| VTICG | 1 | 1 | 560.0 | 695.0 |

Only 4 aircraft were reassigned out of 48 total — validating the model’s **selective recovery** behavior.  
Unnecessary reassignments were avoided, maintaining global schedule stability.

---

## 7. Visualization

<img width="4200" height="1800" alt="image" src="https://github.com/user-attachments/assets/e998266e-1e19-49c1-8067-2abca627690c" />

<img width="3000" height="1500" alt="image" src="https://github.com/user-attachments/assets/c2230206-46a5-4586-b4f9-f866a0248408" />

<img width="2400" height="3000" alt="image" src="https://github.com/user-attachments/assets/ca4921db-0229-4edd-8341-c7fed7fe4485" />

<img width="2400" height="1500" alt="image" src="https://github.com/user-attachments/assets/c5ea5f6b-2d57-40d9-b50a-54e719f9b6f7" />

## Usage instructions
```python

# 1. Create and activate environment
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train HeteroGNN
python -m src.train

# 4. Run inference (to get key aircraft)
python -m src.inference

# 5. Run MILP optimization
python -m src.cplex_optimizer

# 6. Results Visualisation
python -m src.results

