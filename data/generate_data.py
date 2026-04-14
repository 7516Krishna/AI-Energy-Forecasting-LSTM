import pandas as pd
import numpy as np

# Generate hourly data
date_range = pd.date_range(start="2020-01-01", periods=1000, freq="h")
# Simulated energy usage
energy = (
    200 
    + 50 * np.sin(np.arange(1000) / 24)
    + 10 * np.random.randn(1000)
)

df = pd.DataFrame({
    "datetime": date_range,
    "energy": energy
})

df.to_csv("data/energy.csv", index=False)

print("Dataset created successfully!")