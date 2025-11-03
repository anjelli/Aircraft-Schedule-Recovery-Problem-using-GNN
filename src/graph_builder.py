import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
from collections import defaultdict

class DayScheduleGraphBuilder:
    def __init__(self, flights_df, aircraft_df):
        self.flights = flights_df.copy()
        self.aircraft = aircraft_df.copy()

    def build_graph_for_disruption(self, disruption_dict, labels_map=None):
        """
        Robust version — handles empty days, missing columns, single airports, etc.
        """
        day = disruption_dict.get("day")
        disrupted_tail = disruption_dict.get("disrupted_tail")

        # subset flights for the same day
        flights_day = self.flights[self.flights["day"] == day].copy()
        if flights_day.empty:
            raise ValueError(f"No flights found for day {day}")

        # ensure required columns exist
        for c in ["origin", "dest", "tail", "duration", "std_min", "sta_min"]:
            if c not in flights_day.columns:
                flights_day[c] = None

        # --- aircraft nodes ---
        unique_tails = sorted(flights_day["tail"].astype(str).unique().tolist())
        x_aircraft = torch.tensor(
            np.stack([
                flights_day.groupby("tail")["duration"].mean().reindex(unique_tails, fill_value=0).values,
                flights_day.groupby("tail")["std_min"].mean().reindex(unique_tails, fill_value=0).values,
            ], axis=1),
            dtype=torch.float,
        )

        # --- flight nodes ---
        flight_ids = flights_day["flight_id"].astype(str).tolist()
        x_flight = torch.tensor(
            np.stack([
                flights_day["duration"].fillna(0).values,
                flights_day["std_min"].fillna(0).values,
            ], axis=1),
            dtype=torch.float,
        )

        # --- airport nodes ---
        cols = [c for c in ["origin", "dest"] if c in flights_day.columns]
        airport_list = flights_day[cols].astype(str).values.ravel().tolist()
        unique_airports = sorted(set(airport_list)) if len(airport_list) > 0 else []
        if not unique_airports:
            raise ValueError("No airports found in this day's data.")

        x_airport = torch.zeros((len(unique_airports), 2), dtype=torch.float)

        # --- node mappings ---
        tail2aircraft_idx = {t: i for i, t in enumerate(unique_tails)}
        flight2idx = {f: i for i, f in enumerate(flight_ids)}
        airport2idx = {a: i for i, a in enumerate(unique_airports)}

        # --- edges ---
        edges_aircraft_to_flight = [
            [tail2aircraft_idx[row.tail], flight2idx[row.flight_id]]
            for _, row in flights_day.iterrows()
            if row.tail in tail2aircraft_idx and row.flight_id in flight2idx
        ]
        if edges_aircraft_to_flight:
            edge_index_aircraft_to_flight = torch.tensor(edges_aircraft_to_flight, dtype=torch.long).t().contiguous()
        else:
            edge_index_aircraft_to_flight = torch.zeros((2, 0), dtype=torch.long)

        edges_flight_to_airport = []
        for _, row in flights_day.iterrows():
            if row.origin in airport2idx and row.flight_id in flight2idx:
                edges_flight_to_airport.append([flight2idx[row.flight_id], airport2idx[row.origin]])
            if row.dest in airport2idx and row.flight_id in flight2idx:
                edges_flight_to_airport.append([flight2idx[row.flight_id], airport2idx[row.dest]])
        edge_index_flight_to_airport = (
            torch.tensor(edges_flight_to_airport, dtype=torch.long).t().contiguous()
            if edges_flight_to_airport else torch.zeros((2, 0), dtype=torch.long)
        )

                # --- build HeteroData ---
        data = HeteroData()
        data["aircraft"].x = x_aircraft
        data["flight"].x = x_flight
        data["airport"].x = x_airport

        # forward edges
        data["aircraft", "operates", "flight"].edge_index = edge_index_aircraft_to_flight
        data["flight", "connects", "airport"].edge_index = edge_index_flight_to_airport

        if edge_index_flight_to_airport.numel() > 0:
            data["airport", "connected_from", "flight"].edge_index = edge_index_flight_to_airport.flip(0)

        # --- add reverse edges so aircraft receive messages ---
        if edge_index_aircraft_to_flight.numel() > 0:
            edge_index_flight_to_aircraft = edge_index_aircraft_to_flight.flip(0)
        else:
            edge_index_flight_to_aircraft = torch.zeros((2, 0), dtype=torch.long)
        data["flight", "operated_by", "aircraft"].edge_index = edge_index_flight_to_aircraft

        # --- optional: add self-loops on aircraft to stabilize message passing ---
        if len(unique_tails) > 0:
            self_loops = torch.arange(len(unique_tails), dtype=torch.long).unsqueeze(0).repeat(2, 1)
        else:
            self_loops = torch.zeros((2, 0), dtype=torch.long)
        data["aircraft", "connects", "aircraft"].edge_index = self_loops

        mapping = {
            "tail2aircraft_idx": tail2aircraft_idx,
            "flight2idx": flight2idx,
            "airport2idx": airport2idx,
            "aircraft_idx2tail": {v: k for k, v in tail2aircraft_idx.items()},
        }

        return data, mapping


        return data, mapping

