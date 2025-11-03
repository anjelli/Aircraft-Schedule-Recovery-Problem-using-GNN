import pandas as pd
import numpy as np

def preprocess_excel(file_path: str):
    xl = pd.ExcelFile("C:/Users/AJIT SINGH/BTP_sem7/data/Flight Schedule 171F.xlsx")
    sheet = xl.sheet_names[0]
    raw = xl.parse(sheet)

    flights = pd.DataFrame()
    flights["flight_id"] = raw["FLTNO"].astype(str)
    flights["tail"] = raw["TAIL"].astype(str)
    flights["origin"] = raw["DEP"].astype(str)
    flights["dest"] = raw["ARR"].astype(str)

    def to_minutes(dt):
        if pd.isna(dt):
            return np.nan
        ts = pd.to_datetime(dt)
        return ts.hour * 60 + ts.minute

    def day_label(dt):
        if pd.isna(dt):
            return "unknown"
        ts = pd.to_datetime(dt)
        return ts.date().isoformat()

    flights["std_min"] = raw["STDIST"].apply(to_minutes)
    flights["sta_min"] = raw["STAIST"].apply(to_minutes)
    flights["duration"] = (flights["sta_min"] - flights["std_min"]).fillna(0).astype(float)
    flights["day"] = raw["STDIST"].apply(lambda x: day_label(x) if not pd.isna(x) else "unknown")

    flights.to_csv("data/flights_canonical.csv", index=False)

    aircraft = pd.DataFrame()
    aircraft["tail"] = raw["TAIL"].astype(str)
    aircraft["ac_type"] = raw["BODY"].astype(str)
    aircraft.drop_duplicates(subset=["tail"]).to_csv("data/aircraft_canonical.csv", index=False)

    disruptions = []
    for tail in aircraft["tail"].unique():
        sub = flights[flights["tail"] == tail].sort_values("std_min")
        if sub.empty:
            continue
        disruptions.append({
            "disruption_id": f"dis_{tail}",
            "day": sub["day"].iloc[0],
            "disrupted_tail": tail,
            "disruption_time": int(sub["std_min"].iloc[0]),
        })
    pd.DataFrame(disruptions).to_csv("data/disruptions_synthetic.csv", index=False)

if __name__ == "__main__":
    preprocess_excel("data/Flight_Schedule_171F.xlsx")
    print("Preprocessing complete. CSVs saved in data/.")
