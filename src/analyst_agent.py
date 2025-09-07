import json
import logging
import requests
import pandas as pd
import time
import random
from typing import Dict, Any, List
from src.config import config
from src.tools import clean_data, generate_basic_insights

logger = logging.getLogger(__name__)

class AnalystAgent:
    def __init__(self):
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        return """You are an expert data analyst AI. Analyze ANY dataset and provide a structured, professional report."""

    def analyze(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Main analysis method. Sequences automated cleaning and LLM analysis."""
        cleaning_ops = self._determine_cleaning_ops(df)
        cleaned_df, cleaning_log = clean_data(df, cleaning_ops)
        insights = generate_basic_insights(cleaned_df)

        context = {
            "dataset_name": dataset_name,
            "shape": cleaned_df.shape,
            "columns": list(cleaned_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
            "sample_data": cleaned_df.head(3).to_dict(),
            "cleaning_log": cleaning_log,
            "basic_insights": insights
        }

        # Truncate large sample data to avoid token limits
        context = self._truncate_large_data(context)

        prompt = self._build_prompt(context)

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": self.system_prompt + "\n\n" + prompt}]}
            ],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200}
        }

        # Check if payload is too large
        if self._is_payload_too_large(payload):
            logger.warning("Payload too large for API, using fallback analysis")
            analysis_report = self._generate_fallback_analysis(cleaned_df, context)
        else:
            try:
                analysis_report = self._call_google_ai_api(payload, headers)
            except Exception as e:
                logger.error(f"API analysis failed: {e}")
                analysis_report = self._generate_fallback_analysis(cleaned_df, context)

        return {
            "cleaned_data": cleaned_df,
            "cleaning_log": cleaning_log,
            "insights": insights,
            "analysis_report": analysis_report
        }

    def _truncate_large_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate large data to avoid API token limits"""
        context = context.copy()
        
        # Truncate sample data if too large
        if len(str(context.get('sample_data', ''))) > 5000:
            context['sample_data'] = "Sample data too large - showing first 2 rows only"
            if 'cleaned_data' in context:
                context['sample_data'] = context['cleaned_data'].head(2).to_dict()
        
        # Truncate basic insights if too large
        if len(str(context.get('basic_insights', ''))) > 10000:
            context['basic_insights'] = {
                'dataset_overview': context['basic_insights'].get('dataset_overview', {}),
                'missing_values': context['basic_insights'].get('missing_values', {})
            }
        
        return context

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        context_str = json.dumps(context, indent=2, ensure_ascii=False)
        
        # Ensure prompt isn't too long
        if len(context_str) > 15000:
            context_str = context_str[:15000] + "... (truncated due to size)"
        
        return (
            "Analyze this dataset comprehensively. The data has already been cleaned.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            "Provide a structured report with the following sections:\n"
            "1. **Dataset Overview** – summarize domain, size, column types.\n"
            "2. **Data Quality Notes** – quantify missing %, duplicates, format issues.\n"
            "3. **Key Insights** – highlight important stats, trends, correlations, and group comparisons.\n"
            "4. **Recommendations** – actionable next steps or policy suggestions.\n"
            "5. **Suggested Visualizations** – list charts/plots that best represent findings.\n"
        )

    def _call_google_ai_api(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        api_url = f"{config.GOOGLE_API_URL}?key={config.GOOGLE_AI_STUDIO_API_KEY}"
        
        for attempt in range(3):
            try:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                
                response = requests.post(url=api_url, json=payload, headers=headers, timeout=30)
                logger.info(f"Attempt {attempt + 1} - HTTP Status: {response.status_code}")
                
                if response.status_code == 429:
                    logger.warning(f"Rate limited on attempt {attempt + 1}, waiting longer...")
                    time.sleep(10)  # Longer wait for rate limits
                    continue
                
                response.raise_for_status()
                response_data = response.json()
                candidates = response_data.get("candidates", [])
                
                if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                    content = candidates[0]["content"]["parts"][0].get("text", "")
                    if content:
                        return content
                
                raise ValueError("Empty or malformed response from Google AI Studio API.")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Final attempt
                    logger.error("All attempts to call Google AI Studio API failed.")
                    raise Exception("API call failed after 3 attempts")
        
        raise Exception("All attempts to call Google AI Studio API failed.")

    def _is_payload_too_large(self, payload: Dict[str, Any]) -> bool:
        """Estimate if payload exceeds API limits"""
        try:
            payload_str = json.dumps(payload)
            return len(payload_str) > 30000  # Conservative estimate
        except:
            return False

    def _generate_fallback_analysis(self, df: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate basic analysis locally when API fails"""
        insights = context.get('basic_insights', {})
        
        return f"""# Fallback Analysis Report
## Dataset Overview
- **Shape**: {context.get('shape', 'N/A')}
- **Columns**: {', '.join(context.get('columns', []))}
- **Total Rows**: {df.shape[0]}
- **Total Columns**: {df.shape[1]}

## Data Quality Notes
- **Missing Values**: {sum(insights.get('missing_values', {}).values())} total missing values
- **Missing Percentage**: {sum(insights.get('missing_percent', {}).values()):.2f}% overall

## Key Insights (Basic)
- **Numeric Columns**: {len(df.select_dtypes(include=[np.number]).columns)}
- **Categorical Columns**: {len(df.select_dtypes(include=['object', 'category']).columns)}
- **Date Columns**: {len(df.select_dtypes(include=['datetime']).columns)}

## Recommendations
1. **Data Collection**: Ensure consistent data collection practices
2. **Validation**: Implement data validation rules
3. **Monitoring**: Set up ongoing data quality monitoring

## Suggested Visualizations
- Histograms for numeric columns
- Bar charts for categorical distributions
- Correlation heatmap for numeric relationships
"""

    def _determine_cleaning_ops(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        operations = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                operations.append({
                    "action": "fill_na",
                    "column": col,
                    "value": self._suggest_fill_value(df[col])
                })
        if df.duplicated().sum() > 0:
            operations.append({"action": "drop_duplicates"})
        return operations

    def _suggest_fill_value(self, series: pd.Series) -> Any:
        if pd.api.types.is_numeric_dtype(series):
            return series.median()
        return series.mode()[0] if not series.mode().empty else "Unknown"