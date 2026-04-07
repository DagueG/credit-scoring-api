"""Build reference and drift baseline datasets from application data."""

import logging
import os
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_reference_and_baseline(
    csv_path: str = "data/application_train.csv",
    output_dir: str = "data",
    reference_size: int = 1000,
    baseline_size: int = 10000,
    seed: int = 42
) -> None:
    """Build reference and drift baseline datasets.
    
    Args:
        csv_path: Path to the application_train.csv file
        output_dir: Output directory for parquet files
        reference_size: Number of samples for reference dataset
        baseline_size: Number of samples for drift baseline
        seed: Random seed for reproducibility
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the data
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} rows with {df.shape[1]} columns")
    
    # Get SK_ID_CURR (client ID)
    if 'SK_ID_CURR' not in df.columns:
        raise ValueError("DataFrame must contain 'SK_ID_CURR' column")
    
    # Remove TARGET column for reference dataset
    features_df = df.drop(columns=['TARGET'], errors='ignore')
    
    # Build reference dataset (1000 clients with SK_ID_CURR)
    logger.info(f"Building reference dataset with {reference_size} samples")
    reference_df = features_df.sample(n=reference_size, random_state=seed)
    reference_df = reference_df.set_index('SK_ID_CURR')  # Index for fast lookup
    
    reference_path = os.path.join(output_dir, 'clients_reference.parquet')
    reference_df.to_parquet(reference_path)
    logger.info(f"Saved reference dataset to {reference_path} ({len(reference_df)} rows)")
    
    # Build drift baseline (10000 clients without SK_ID_CURR)
    logger.info(f"Building drift baseline with {baseline_size} samples")
    # Use all features except SK_ID_CURR for drift detection
    feature_cols = [col for col in features_df.columns if col != 'SK_ID_CURR']
    drift_df = features_df[feature_cols].sample(n=baseline_size, random_state=seed)
    
    drift_path = os.path.join(output_dir, 'drift_baseline.parquet')
    drift_df.to_parquet(drift_path)
    logger.info(f"Saved drift baseline to {drift_path} ({len(drift_df)} rows)")
    
    logger.info("✓ Reference and baseline datasets created successfully")


if __name__ == "__main__":
    build_reference_and_baseline()
