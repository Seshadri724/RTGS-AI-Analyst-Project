import os
import json
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")


def save_artifact(data, filename, output_dir):
    """Save artifacts (DataFrame, text, dict, etc.) into the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, (str, dict, list)):
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f, indent=4)
    else:
        raise ValueError(f"Unsupported data type for {filename}: {type(data)}")
    
    return filepath


def clean_data(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    """Generic data cleaning function"""
    df_clean = df.copy()
    log_entries = []

    for op in operations:
        try:
            action = op.get("action")
            column = op.get("column")

            if action == "fill_na" and column:
                value = op.get("value", 0)
                df_clean[column] = df_clean[column].fillna(value)
                log_entries.append(f"Filled missing values in '{column}' with {value}")

            elif action == "drop_na" and column:
                df_clean.dropna(subset=[column], inplace=True)
                log_entries.append(f"Dropped rows with missing values in '{column}'")

            elif action == "drop_duplicates":
                df_clean.drop_duplicates(inplace=True)
                log_entries.append("Dropped duplicate rows")

            elif action == "convert_type" and column:
                new_type = op.get("new_type", "string")
                if new_type == "numeric":
                    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                elif new_type == "datetime":
                    df_clean[column] = pd.to_datetime(df_clean[column], errors='coerce')
                log_entries.append(f"Converted '{column}' to {new_type}")

        except Exception as e:
            logger.error(f"Error executing operation {op}: {e}")

    return df_clean, log_entries


def generate_basic_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate richer insights from any DataFrame"""
    insights = {
        "dataset_overview": {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "summary_statistics": df.describe(include='all').to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
        "sample_data": df.head(5).to_dict()
    }

    # Correlations
    if not df.select_dtypes(include=[np.number]).empty:
        insights["correlations"] = df.corr(numeric_only=True).round(3).to_dict()

    # Group-by summaries (if categorical + numeric)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    groupby_insights = {}
    for cat_col in categorical_cols:
        groupby_insights[cat_col] = df.groupby(cat_col)[numeric_cols].mean().round(2).to_dict()
    if groupby_insights:
        insights["groupby_insights"] = groupby_insights

    # Ranges for numeric
    ranges = {}
    for col in numeric_cols:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "range": float(df[col].max() - df[col].min())
        }
    if ranges:
        insights["numeric_ranges"] = ranges

    return insights
