import pandas as pd
import numpy as np
import logging
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationManager:
    def __init__(self):
        self.previous_results = {}
    
    def validate_cleaning_operation(self, df_before: pd.DataFrame, df_after: pd.DataFrame, 
                                  operation: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that cleaning operations actually improved data quality"""
        try:
            action = operation.get("action")
            column = operation.get("column", "")
            
            if action == "fill_na" and column:
                nulls_before = df_before[column].isnull().sum()
                nulls_after = df_after[column].isnull().sum()
                if nulls_after >= nulls_before:
                    return False, f"Fill operation failed for {column}: {nulls_before} -> {nulls_after} nulls"
                return True, f"Fill operation successful: {nulls_before} -> {nulls_after} nulls"
            
            elif action == "drop_na" and column:
                nulls_before = df_before[column].isnull().sum()
                rows_before = len(df_before)
                rows_after = len(df_after)
                expected_removed = nulls_before
                actual_removed = rows_before - rows_after
                
                if actual_removed < expected_removed * 0.8:  # Allow some flexibility
                    return False, f"Drop operation failed for {column}: expected ~{expected_removed}, got {actual_removed}"
                return True, f"Drop operation successful: removed {actual_removed} rows"
            
            elif action == "drop_duplicates":
                duplicates_before = df_before.duplicated().sum()
                duplicates_after = df_after.duplicated().sum()
                if duplicates_after >= duplicates_before:
                    return False, f"Drop duplicates failed: {duplicates_before} -> {duplicates_after}"
                return True, f"Drop duplicates successful: {duplicates_before} -> {duplicates_after}"
            
            return True, f"Operation {action} validated"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_insights(self, insights: Dict[str, Any], df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that insights make sense and match the actual data"""
        errors = []
        
        # Check shape matches
        if insights["dataset_overview"]["shape"] != df.shape:
            errors.append(f"Shape mismatch: insights {insights['dataset_overview']['shape']} vs actual {df.shape}")
        
        # Check columns match
        insight_cols = set(insights["dataset_overview"]["columns"])
        actual_cols = set(df.columns)
        if insight_cols != actual_cols:
            errors.append(f"Columns mismatch: {insight_cols.symmetric_difference(actual_cols)}")
        
        # Check missing values calculation
        actual_missing = df.isnull().sum().to_dict()
        reported_missing = insights["missing_values"]
        for col in actual_missing:
            if col in reported_missing and actual_missing[col] != reported_missing[col]:
                errors.append(f"Missing values mismatch for {col}: {actual_missing[col]} vs {reported_missing[col]}")
        
        return len(errors) == 0, errors

    def detect_hallucinations(self, analysis_report: str, actual_insights: Dict[str, Any]) -> List[str]:
        """Check if LLM is making up information not supported by data"""
        warnings = []
        report_lower = analysis_report.lower()
        
        # Check for fabricated statistics
        fabricated_patterns = [
            r"exactly (\d+\.\d+)%", 
            r"precisely (\d+)%",
            r"definitely shows that",
            r"without a doubt",
            r"certainly proves",
            r"clearly demonstrates"
        ]
        
        for pattern in fabricated_patterns:
            matches = re.findall(pattern, report_lower)
            if matches:
                warnings.append(f"Potential hallucination pattern: '{pattern}' found")
        
        # Check for claims about data that doesn't exist
        if "correlation" in report_lower and "correlations" not in actual_insights:
            warnings.append("Mentioned correlations but no correlation data in insights")
        
        if "trend" in report_lower and len(actual_insights.get("dataset_overview", {}).get("columns", [])) < 2:
            warnings.append("Mentioned trends but insufficient columns for trend analysis")
        
        return warnings

    def check_consistency(self, current_results: Dict[str, Any], dataset_name: str) -> List[str]:
        """Compare with previous runs to detect anomalies"""
        warnings = []
        
        if dataset_name in self.previous_results:
            previous = self.previous_results[dataset_name]
            current = current_results
            
            # Check data shape consistency
            if current["cleaned_data"].shape != previous["cleaned_data"].shape:
                warnings.append(f"Shape inconsistency: {previous['cleaned_data'].shape} -> {current['cleaned_data'].shape}")
            
            # Check cleaning log consistency
            current_logs = len(current.get("cleaning_log", []))
            previous_logs = len(previous.get("cleaning_log", []))
            if abs(current_logs - previous_logs) > max(current_logs, previous_logs) * 0.3:
                warnings.append(f"Cleaning log count variation: {previous_logs} -> {current_logs}")
        
        # Store current results for future comparison
        self.previous_results[dataset_name] = current_results
        
        return warnings

    def create_messy_dataset(self, df: pd.DataFrame, messiness_level: float = 0.3) -> pd.DataFrame:
        """Create intentionally messy data for stress testing"""
        messy_df = df.copy()
        
        # Remove random rows
        messy_df = messy_df.sample(frac=1 - messiness_level * 0.5)
        
        # Add missing values to numeric columns
        numeric_cols = messy_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mask = np.random.random(len(messy_df)) < messiness_level * 0.3
            messy_df.loc[mask, col] = np.nan
        
        # Add missing values to categorical columns
        categorical_cols = messy_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mask = np.random.random(len(messy_df)) < messiness_level * 0.2
            messy_df.loc[mask, col] = None
        
        # Add duplicates
        if len(messy_df) > 10:
            duplicates = messy_df.head(max(2, int(len(messy_df) * messiness_level * 0.2)))
            messy_df = pd.concat([messy_df, duplicates], ignore_index=True)
        
        # Add some extreme values to numeric columns
        for col in numeric_cols:
            if len(messy_df) > 5:
                extreme_idx = messy_df.sample(n=min(3, len(messy_df)//10)).index
                messy_df.loc[extreme_idx, col] = messy_df[col].max() * 100
        
        return messy_df

    def run_cross_dataset_test(self, dataset_paths: Dict[str, Path]) -> Dict[str, List[str]]:
        """Test the system with multiple domain datasets"""
        results = {}
        
        for domain, path in dataset_paths.items():
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    # Basic validation that dataset can be processed
                    if len(df) > 0 and len(df.columns) > 0:
                        results[domain] = ["✅ Dataset loaded successfully"]
                    else:
                        results[domain] = ["❌ Empty dataset"]
                except Exception as e:
                    results[domain] = [f"❌ Failed to load: {str(e)}"]
            else:
                results[domain] = ["❌ Dataset file not found"]
        
        return results