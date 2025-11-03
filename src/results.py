import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_assignment(flights_df, assigned, key_tails, title="Reassignment Visualization"):
    # Merge assignment results into flights data
    df = flights_df.copy()
    df['assigned_tail'] = df['flight_id'].map(assigned).fillna(df['tail'])
    df['reassigned'] = df['assigned_tail'] != df['tail']

    # --- Summary
    total = len(df)
    reassigned = df['reassigned'].sum()
    print(f"Total flights: {total}")
    print(f"Affected flights (reassigned): {reassigned}")
    print(f"Unchanged flights: {total - reassigned}")
    print("Affected counts per key tail:")
    print(df[df['reassigned']]['assigned_tail'].value_counts().to_dict())

    # --- Aggregate summary table
    summary = df.groupby('assigned_tail').agg(
        total_flights=('flight_id', 'count'),
        reassigned=('reassigned', 'sum'),
        avg_std=('std_min', 'mean'),
        avg_sta=('sta_min', 'mean')
    ).reset_index()
    print("\nSummary Table:")
    print(summary)

    # --- Visualization (Gantt-style timeline)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {'key_tail': '#ff7f0e', 'fixed_tail': '#1f77b4'}
    y_positions = {tail: i for i, tail in enumerate(sorted(df['assigned_tail'].unique()))}

    for _, row in df.iterrows():
        tail = row['assigned_tail']
        color = colors['key_tail'] if tail in key_tails else colors['fixed_tail']
        ax.barh(y_positions[tail],
                row['sta_min'] - row['std_min'],
                left=row['std_min'],
                height=0.6,
                color=color,
                alpha=0.8)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel("Minutes since start of day")
    ax.set_title(title)

    legend = [mpatches.Patch(color=v, label=k.replace('_', ' ').title()) for k, v in colors.items()]
    ax.legend(handles=legend, loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    flights = pd.read_csv("data/flights_canonical.csv")

    # Example assignment result (replace with your optimizer’s output)
    assigned = {
        'ZZ0640': 'VTBBL',
        'ZZ0639': 'VTAIH',
        'ZZ0627': 'VTAIH',
        'ZZ0628': 'VTBBL',
        'ZZ0549': 'VTBBP',
        'ZZ0550': 'VTICG',
        'ZZ0516': 'VTBBP',
        'ZZ0515': 'VTBBP'
    }
    key_tails = ["VTAIH", "VTBBL", "VTBBP", "VTICG"]

    visualize_assignment(flights, assigned, key_tails)
