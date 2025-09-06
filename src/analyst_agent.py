import json
import logging
import requests
import pandas as pd
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

        prompt = self._build_prompt(context)

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": self.system_prompt + "\n\n" + prompt}]}
            ],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200}
        }

        analysis_report = self._call_google_ai_api(payload, headers)

        return {
            "cleaned_data": cleaned_df,
            "cleaning_log": cleaning_log,
            "insights": insights,
            "analysis_report": analysis_report
        }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        context_str = json.dumps(context, indent=2, ensure_ascii=False)
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
                response = requests.post(url=api_url, json=payload, headers=headers)
                logger.info(f"Attempt {attempt + 1} - HTTP Status: {response.status_code}")
                response.raise_for_status()
                response_data = response.json()
                candidates = response_data.get("candidates", [])
                if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                    content = candidates[0]["content"]["parts"][0].get("text", "")
                    if content:
                        return content
                raise ValueError("Empty or malformed response from Google AI Studio API.")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        logger.error("All attempts to call Google AI Studio API failed.")
        return "Error: Could not generate analysis report from Google AI Studio API."

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
