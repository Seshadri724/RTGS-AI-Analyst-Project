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
        return """You are an expert data analyst AI. Analyze ANY dataset with this structure:

1. DATA UNDERSTANDING: Comprehend the dataset's domain, columns, types
2. DATA CLEANING: Handle missing values, duplicates, format standardization
3. INSIGHT GENERATION: Provide statistics, trends, patterns, correlations
4. RECOMMENDATIONS: Suggest actionable insights for policymakers

Be thorough, detailed, and structured in your analysis."""

    def analyze(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Main analysis method. Sequences automated cleaning and LLM analysis."""

        # Step 1: Automated cleaning
        cleaning_ops = self._determine_cleaning_ops(df)
        cleaned_df, cleaning_log = clean_data(df, cleaning_ops)

        # Step 2: Generate basic insights
        insights = generate_basic_insights(cleaned_df)

        # Step 3: Prepare context and prompt
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

        # Step 4: Call the Google AI Studio API
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": self.system_prompt + "\n\n" + prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000
            }
        }

        analysis_report = self._call_google_ai_api(payload, headers)

        return {
            "cleaned_data": cleaned_df,
            "cleaning_log": cleaning_log,
            "insights": insights,
            "analysis_report": analysis_report
        }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Constructs the user prompt for LLM analysis."""
        context_str = json.dumps(context, indent=2, ensure_ascii=False)
        return (
            "Analyze this dataset comprehensively. The data has already been cleaned.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            "Provide a detailed report including:\n"
            "1. Data quality assessment (based on the cleaning log)\n"
            "2. Key insights and patterns (expand on the provided basic insights)\n"
            "3. Policy recommendations\n"
            "4. Suggested visualizations"
        )

    def _call_google_ai_api(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        """Handles API interaction with retries and error handling."""
        api_url = f"{config.GOOGLE_API_URL}?key={config.GOOGLE_AI_STUDIO_API_KEY}"
        for attempt in range(3):
            try:
                response = requests.post(
                    url=api_url,
                    json=payload,
                    headers=headers
                )
                logger.info(f"Attempt {attempt + 1} - HTTP Status: {response.status_code}")
                logger.info(f"Attempt {attempt + 1} - Raw Response: {response.text}")
                response.raise_for_status()
                response_data = response.json()
                logger.info(f"Attempt {attempt + 1} - Parsed Response: {json.dumps(response_data, indent=2)}")
                candidates = response_data.get("candidates", [])
                if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                    content = candidates[0]["content"]["parts"][0].get("text", "")
                    if content:
                        return content
                raise ValueError("Empty or malformed response from Google AI Studio API.")
            except (requests.exceptions.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        logger.error("All attempts to call Google AI Studio API failed.")
        return "Error: Could not generate analysis report from Google AI Studio API."

    def _determine_cleaning_ops(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Determines cleaning operations based on missing values and duplicates."""
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
        """Suggests appropriate fill value based on column type."""
        if pd.api.types.is_numeric_dtype(series):
            return series.median()
        return series.mode()[0] if not series.mode().empty else "Unknown"