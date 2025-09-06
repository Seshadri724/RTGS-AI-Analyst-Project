import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Optional
import logging
from src.config import config

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

def save_artifact(data, filename: str) -> str:
    """Save any artifact to file"""
    filepath = os.path.join(config.ARTIFACTS_DIR, filename)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
    elif isinstance(data, (dict, list)):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath

def clean_data(df: pd.DataFrame, operations: List[Dict]) -> pd.DataFrame:
    """Generic data cleaning function"""
    df_clean = df.copy()
    log_entries = []
    
    for op in operations:
        try:
            action = op.get("action")
            column = op.get("column")
            
            if action == "fill_na" and column:
                value = op.get("value", 0)
                df_clean[column].fillna(value, inplace=True)
                log_entries.append(f"Filled missing values in '{column}' with {value}")
                
            elif action == "drop_na" and column:
                df_clean.dropna(subset=[column], inplace=True)
                log_entries.append(f"Dropped rows with missing values in '{column}'")
                
            elif action == "drop_duplicates":
                df_clean.drop_duplicates(inplace=True)
                log_entries.append("Dropped duplicate rows")
                
            elif action == "convert_type" and column:
                new_type = op.get("new_type", "string")
                try:
                    if new_type == "numeric":
                        df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                    elif new_type == "datetime":
                        df_clean[column] = pd.to_datetime(df_clean[column], errors='coerce')
                    log_entries.append(f"Converted '{column}' to {new_type}")
                except:
                    pass
                        
        except Exception as e:
            logger.error(f"Error executing operation {op}: {e}")
    
    return df_clean, log_entries

def generate_basic_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate basic insights from any DataFrame"""
    insights = {
        "dataset_overview": {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "summary_statistics": df.describe(include='all').to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict()
    }
    return insights