# src/pulp_optimizer_fixed.py
import pandas as pd
import numpy as np
import itertools
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PulpSolverError, LpStatus

def to_minutes_safe(col):
    """Convert datetime-like column to minutes since midnight; or pass through numeric."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float).fillna(0.0)
    parsed = pd.to_datetime(col, errors='coerce')
    mins = (parsed.dt.hour.fillna(0) * 60) + parsed.dt.minute.fillna(0)
    return mins.astype(float).fillna(0.0)

def optimize_assignment_reassign_keys(flights_df,
                                      key_tails,
                                      allowed_tails=None,
                                      probs=None,
                                      min_turnaround=30,
                                      allow_unassigned_penalty=1000):
    """
    Reassign only flights that currently belong to key_tails (GNN output).
    - flights_df: must contain columns ['flight_id','tail','std_min','sta_min'] (std_min/sta_min may be datetimes)
    - key_tails: list of tails chosen by GNN (these flights are candidates for reassignment)
    - allowed_tails: list of tails allowed as targets (default: all tails present in flights_df)
    - probs: optional dict tail -> probability (higher means preferred)
    - min_turnaround: minutes buffer required between consecutive flights on same tail
    - allow_unassigned_penalty: penalty cost for leaving an affected flight unassigned (keeps problem feasible)
    Returns:
      assigned dict mapping flight_id -> tail (original tails kept for non-affected flights).
    """

    # 1) Validate / normalize times
    df = flights_df.copy()
    if 'std_min' not in df.columns or 'sta_min' not in df.columns:
        raise ValueError("flights_df must contain 'std_min' and 'sta_min' columns")
    df['std_min'] = to_minutes_safe(df['std_min'])
    df['sta_min'] = to_minutes_safe(df['sta_min'])
    df = df.reset_index(drop=True)

    # original assignment map
    orig_assign = dict(zip(df['flight_id'], df['tail']))

    # allowed target tails default
    all_tails = sorted(df['tail'].unique().tolist())
    if allowed_tails is None:
        allowed_tails = all_tails
    allowed_tails = [t for t in allowed_tails if t in all_tails]

    # affected flights = those currently assigned to any of key_tails
    affected_mask = df['tail'].isin(key_tails)
    affected_flights = df.loc[affected_mask, 'flight_id'].tolist()
    fixed_flights = df.loc[~affected_mask, ['flight_id', 'tail', 'std_min', 'sta_min']].copy()

    print(f"Total flights: {len(df)}")
    print(f"Affected flights (to reassign): {len(affected_flights)}")
    print(f"Fixed flights (kept as-is): {len(fixed_flights)}")
    print("Affected counts per key tail:", df.loc[affected_mask].groupby('tail').size().to_dict())

    # quick lookup rows
    flight_rows = {row['flight_id']: row for _, row in df.iterrows()}

    # 2) Build MILP (PuLP)
    prob = LpProblem("Reassign_Key_Aircraft", LpMinimize)

    # variables: x[(f,t)] only for affected flights and allowed tails
    x = {}
    for f in affected_flights:
        for t in allowed_tails:
            x[(f,t)] = LpVariable(f"x_{f}_{t}", cat="Binary")

    # slack/unassigned variables u[f]
    u = {f: LpVariable(f"u_unassigned_{f}", lowBound=0, upBound=1) for f in affected_flights}

    # objective: prefer tails with higher prob (lower cost), penalize unassigned strongly
    def tail_cost(t):
        if probs is None:
            return 1.0
        return max(0.0, 1.0 - float(probs.get(t, 0.0)))
    prob += lpSum(tail_cost(t) * x[(f,t)] for f in affected_flights for t in allowed_tails) \
            + allow_unassigned_penalty * lpSum(u[f] for f in affected_flights)

    # 3) Constraint: each affected flight assigned to exactly one tail OR left unassigned
    for f in affected_flights:
        prob += lpSum(x[(f,t)] for t in allowed_tails) + u[f] == 1, f"AssignOrLeave_{f}"

    # 4) Forbid assigning an affected flight to a tail if it overlaps with that tail's fixed flights
    constraint_counter = 0
    for _, row in fixed_flights.iterrows():
        f_fixed = row['flight_id']
        tail_fixed = row['tail']
        s_fixed = float(row['std_min'])
        e_fixed = float(row['sta_min'])
        for f in affected_flights:
            r = flight_rows[f]
            s_f = float(r['std_min'])
            e_f = float(r['sta_min'])
            overlap = not ((e_fixed + min_turnaround <= s_f) or (e_f + min_turnaround <= s_fixed))
            if overlap and (f, tail_fixed) in x:
                prob += x[(f, tail_fixed)] <= 0, f"forbid_fixed_overlap_{constraint_counter}"
                constraint_counter += 1


    # 5) No overlap among affected flights assigned to same tail
    for a in allowed_tails:
        # iterate combinations to avoid duplicates
        for f1, f2 in itertools.combinations(affected_flights, 2):
            r1 = flight_rows[f1]; r2 = flight_rows[f2]
            s1, e1 = float(r1['std_min']), float(r1['sta_min'])
            s2, e2 = float(r2['std_min']), float(r2['sta_min'])
            if s1 == s2 and e1 == e2:
                continue
            overlap = not ((e1 + min_turnaround <= s2) or (e2 + min_turnaround <= s1))
            if overlap:
                prob += x[(f1,a)] + x[(f2,a)] <= 1, f"NoOverlap_{a}_{f1}_{f2}"

    # Diagnostics
    n_vars = len(x) + len(u)
    n_cons = len(prob.constraints)
    print(f"Model built: vars={n_vars}, constraints={n_cons}, allowed_tails={len(allowed_tails)}")

    # 6) Solve
    try:
        prob.solve()
    except PulpSolverError as e:
        print("Solver exception:", e)
        return {f: None for f in df['flight_id'].tolist()}

    print("Solver status:", LpStatus[prob.status])

    # 7) Extract assignments: start from original then overwrite affected flights
    assigned = {fid: orig_assign[fid] for fid in df['flight_id'].tolist()}
    for f in affected_flights:
        assigned[f] = None
        # if unassigned slack used
        if u[f].value() is not None and u[f].value() >= 0.5:
            assigned[f] = None
            continue
        for t in allowed_tails:
            val = x[(f,t)].value()
            if val is not None and val > 0.5:
                assigned[f] = t
                break

    # Summary outputs
    assigned_counts = {}
    for fid, tail in assigned.items():
        assigned_counts.setdefault(tail, 0)
        assigned_counts[tail] += 1
    print("Assigned counts (tail -> #flights) sample:", dict(list(assigned_counts.items())[:10]))

    # return full mapping
    return assigned

# -------------------------
# Demo main
# -------------------------
if __name__ == "__main__":
    flights = pd.read_csv("data/flights_canonical.csv")

    # Example: use the 4 key tails from your GNN earlier
    key_tails = ["VTAIH", "VTBBL", "VTBBP", "VTICG"]
    probs = {"VTAIH": 0.0649, "VTBBL": 0.0704, "VTBBP": 0.0798, "VTICG": 0.0586}

    assigned = optimize_assignment_reassign_keys(flights_df=flights,
                                                 key_tails=key_tails,
                                                 allowed_tails=None,
                                                 probs=probs,
                                                 min_turnaround=30,
                                                 allow_unassigned_penalty=1000)

    # Print only affected flights result for readability
    print("\nAffected flights assignments (only those that were considered for reassignment):")
    for f in flights.loc[flights['tail'].isin(key_tails), 'flight_id'].tolist():
        print(f"{f} -> {assigned.get(f)}")
