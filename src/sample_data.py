from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_properties(output_path: str | Path, *, n_samples: int = 500, seed: int = 42) -> Path:
    """Generate synthetic property dataset for demo purposes."""

    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    locations = ["Downtown", "Suburbs", "Waterfront", "Historic", "Hillside"]
    
    data = {
        "sqft": rng.integers(800, 5000, n_samples),
        "bedrooms": rng.integers(1, 6, n_samples),
        "bathrooms": rng.integers(1, 5, n_samples) + rng.random(n_samples),
        "location": rng.choice(locations, n_samples),
        "age": rng.integers(0, 100, n_samples),
        "garage": rng.integers(0, 4, n_samples),
        "pool": rng.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "basement": rng.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }

    df = pd.DataFrame(data)

    base_price = 150000
    sqft_factor = df["sqft"] * 80
    bedroom_factor = df["bedrooms"] * 20000
    bathroom_factor = df["bathrooms"] * 15000
    garage_factor = df["garage"] * 10000
    pool_factor = df["pool"] * 25000
    basement_factor = df["basement"] * 15000
    
    age_factor = -df["age"] * 500
    location_factor = df["location"].map({
        "Downtown": 50000,
        "Waterfront": 100000,
        "Historic": 30000,
        "Hillside": 20000,
        "Suburbs": 10000,
    })

    df["price"] = (
        base_price
        + sqft_factor
        + bedroom_factor
        + bathroom_factor
        + garage_factor
        + pool_factor
        + basement_factor
        + age_factor
        + location_factor
        + rng.normal(0, 20000, n_samples)
    ).astype(int)

    df["price"] = df["price"].abs()
    df = df[df["price"] > 50000]

    df.to_csv(output_path, index=False)
    return output_path
