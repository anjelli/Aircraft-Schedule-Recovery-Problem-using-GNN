# Aircraft-Schedule-Recovery-Problem-using-GNN

# Disruption-Aware Aircraft Reassignment using Heterogeneous Graph Neural Networks and MILP Optimization

This repository contains the implementation for a two-phase framework combining **graph-based disruption prediction** with **optimization-based flight reassignment**.  
The work models airline operations as a dynamic heterogeneous network to identify *key aircraft* affected by potential disruptions and perform *targeted schedule recovery* using mixed-integer linear programming (MILP).

---

## ✈️ Overview

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

## 🧩 Architecture

