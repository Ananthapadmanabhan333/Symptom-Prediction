import numpy as np
import pandas as pd
import os

def generate_fake_data(n=500, save_path="data/fake_data.csv"):
    np.random.seed(42)
    df = pd.DataFrame({
        "sleep_hours": np.random.normal(6, 1.5, n),
        "activity_level": np.random.normal(50, 15, n),
        "screen_time": np.random.normal(5, 2, n),
        "mood_score": np.random.randint(0, 2, n)  # 0 = stable, 1 = depressive episode
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def load_data(path="data/fake_data.csv"):
    return pd.read_csv(path)
