import pandas as pd


def check_network_edges(file_path):
    df = pd.read_csv(file_path)
    edge_traffic = df.groupby('edge_id')['id'].nunique().sort_values(ascending=False)
    a = 0
    for edge, count in edge_traffic.items():
        a = count
        break

    total_time_seconds = df['time'].max()
    h = total_time_seconds / 3600

    print(f"吞吐量为: {a / h:.2f} veh/h")

    mean_speed_ms = df['speed'].mean()
    std_speed_ms = df['speed'].std()

    mean_speed_kmh = mean_speed_ms * 3.6
    std_speed_kmh = std_speed_ms * 3.6

    print(f"平均速度:     {mean_speed_ms:.2f} m/s  (约 {mean_speed_kmh:.2f} km/h)")
    print(f"速度标准差:   {std_speed_ms:.2f} m/s  (约 {std_speed_kmh:.2f} km/h)")

if __name__ == "__main__":
    data_file = "data/merge_20260518-1353091779083589.299307-0_emission.csv"
    check_network_edges(data_file)