import json
import logging
import requests
import pandas as pd
import time
import random
import re
import numpy as np
from typing import Dict, Any, List, Tuple
from src.config import config
from src.tools import clean_data, generate_basic_insights
from src.validation_manager import ValidationManager

logger = logging.getLogger(__name__)

class AnalystAgent:
    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.validation_manager = ValidationManager()

    def _create_system_prompt(self) -> str:
        return """You are an expert data analyst AI. Analyze the dataset and provide a structured, professional report.

CRITICAL INSTRUCTIONS:
1. **ONLY REPORT WHAT THE DATA SHOWS** - Never assume, infer, or make up information
2. **BE SPECIFIC ABOUT LIMITATIONS** - If the data doesn't support certain analyses, say so explicitly
3. **USE QUALIFIED LANGUAGE** - Instead of "proves" use "suggests", instead of "always" use "often"
4. **CITE DATA SOURCES** - Reference specific columns and statistics from the provided context
5. **AVOID ABSOLUTES** - No "all", "every", "never", "always" - use "many", "some", "tends to"
6. **FLAG UNCERTAINTY** - Explicitly state when conclusions are tentative or based on limited data

EXAMPLE OF BAD RESPONSE: "The data proves that all customers prefer product X"
EXAMPLE OF GOOD RESPONSE: "The data suggests many customers in the sample prefer product X (65%), but this is based on limited demographic information"

If you cannot make a confident statement due to data limitations, say: "The available data does not support strong conclusions about [topic]"
"""

    def _filter_context_for_safety(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or modify context elements that often cause hallucinations"""
        safe_context = context.copy()
        
        # Remove very small sample data that might lead to overgeneralization
        if safe_context.get('shape', (0, 0))[0] < 50:  # Small dataset
            safe_context['sample_data'] = "Dataset too small for detailed examples"
        
        # Simplify correlations for small datasets
        insights = safe_context.get('basic_insights', {})
        if insights.get('dataset_overview', {}).get('shape', (0, 0))[0] < 100:
            if 'correlations' in insights:
                insights['correlations'] = "Correlation analysis limited by small sample size"
        
        # Add data quality warnings to context
        missing_percent = insights.get('missing_percent', {})
        high_missing_cols = {col: pct for col, pct in missing_percent.items() if pct > 20}
        
        if high_missing_cols:
            safe_context['data_quality_warnings'] = {
                'high_missing_values': high_missing_cols,
                'message': 'Conclusions involving these columns should be treated with caution due to high missing data rates'
            }
        
        return safe_context

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the dataset"""
        shape = df.shape
        missing_percent = (df.isnull().mean() * 100).round(2).to_dict()
        duplicate_rows = df.duplicated().sum()
        
        quality_score = 10.0
        major_limitations = []
        recommended_caution = "Proceed with standard analysis"
        
        if shape[0] < 100:
            quality_score -= 2.0
            major_limitations.append("Small sample size")
            recommended_caution = "Use caution with generalizability"
        if any(pct > 20 for pct in missing_percent.values()):
            quality_score -= 1.5
            major_limitations.append("High missing data rates")
            recommended_caution = "Validate findings with additional data"
        if duplicate_rows > 0:
            quality_score -= 1.0
            major_limitations.append("Duplicate entries detected")
            recommended_caution = "Review for data integrity issues"
        
        return {
            'sample_size': shape[0],
            'quality_score': max(1.0, quality_score),
            'major_limitations': major_limitations,
            'recommended_caution': recommended_caution
        }

    def _post_process_analysis(self, analysis_report: str, insights: Dict[str, Any]) -> str:
        """Clean up common hallucination patterns in the final report"""
        
        # Remove absolute statements
        absolute_patterns = [
            (r'\b(all|every|each|always|never|none)\b', 'many'),
            (r'\b(proves|definitely|certainly|undoubtedly)\b', 'suggests'),
            (r'\b(perfect|complete|total|absolute)\b', 'substantial')
        ]
        
        for pattern, replacement in absolute_patterns:
            analysis_report = re.sub(pattern, replacement, analysis_report, flags=re.IGNORECASE)
        
        # Add data quality disclaimer
        disclaimer = """
        
## Data Quality Disclaimer
This analysis is based on the provided dataset and may be limited by:
- Sample size and representativeness
- Data completeness and accuracy
- Measurement limitations in the original data collection

Conclusions should be validated with additional data and domain expertise.
"""
        
        return analysis_report + disclaimer

    def detect_hallucinations(self, analysis_report: str, actual_insights: Dict[str, Any]) -> List[str]:
        """More sophisticated hallucination detection"""
        warnings = []
        report_lower = analysis_report.lower()
        
        # 1. Check for statistical impossibilities
        stats = actual_insights.get('summary_statistics', {})
        for col, col_stats in stats.items():
            if 'count' in col_stats:
                # Check for percentages that don't make sense
                percent_matches = re.findall(rf'{col}.*?(\d+\.?\d*)%', report_lower)
                for match in percent_matches:
                    try:
                        percent = float(match)
                        if percent > 100 or percent < 0:
                            warnings.append(f"Impossible percentage for {col}: {percent}%")
                    except:
                        pass
        
        # 2. Check for claims about non-existent correlations
        if 'correlations' not in actual_insights:
            correlation_phrases = ['correlation', 'relationship between', 'associated with']
            for phrase in correlation_phrases:
                if phrase in report_lower:
                    warnings.append(f"Mentioned {phrase} but no correlation data available")
        
        # 3. Check for overgeneralization from small samples
        shape = actual_insights.get('dataset_overview', {}).get('shape', (0, 0))
        if shape[0] < 100:  # Small sample
            generalization_phrases = ['all', 'every', 'always', 'never']
            for phrase in generalization_phrases:
                if phrase in report_lower:
                    warnings.append(f"Overgeneralization from small sample: '{phrase}'")
        
        return warnings

    def _calculate_confidence_score(self, analysis_report: str, insights: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and report content"""
        score = 10.0  # Start with perfect score
        
        # Penalize for small sample size
        shape = insights.get('dataset_overview', {}).get('shape', (0, 0))
        if shape[0] < 100:
            score -= 2.0
        elif shape[0] < 30:
            score -= 4.0
        
        # Penalize for high missing data
        missing_percent = insights.get('missing_percent', {})
        high_missing = any(pct > 20 for pct in missing_percent.values())
        if high_missing:
            score -= 1.5
        
        # Penalize for absolute language
        absolute_words = ['all', 'every', 'always', 'never', 'proves', 'definitely']
        for word in absolute_words:
            if word in analysis_report.lower():
                score -= 0.5
        
        return max(1.0, score)  # Don't go below 1

    def analyze(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Main analysis method with validation."""
        # Store original for validation
        df_original = df.copy()
        
        cleaning_ops = self._determine_cleaning_ops(df)
        cleaned_df, cleaning_log = clean_data(df, cleaning_ops)
        
        # Validate cleaning operations
        validation_results = []
        for op in cleaning_ops:
            valid, message = self.validation_manager.validate_cleaning_operation(df_original, cleaned_df, op)
            validation_results.append(f"{'✅' if valid else '❌'} {message}")
        
        insights = generate_basic_insights(cleaned_df)
        
        # Validate insights
        insights_valid, insights_errors = self.validation_manager.validate_insights(insights, cleaned_df)
        if not insights_valid:
            logger.warning(f"Insights validation failed: {insights_errors}")

        context = {
            "dataset_name": dataset_name,
            "shape": cleaned_df.shape,
            "columns": list(cleaned_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
            "sample_data": cleaned_df.head(3).to_dict(),
            "cleaning_log": cleaning_log,
            "basic_insights": insights,
            "validation_results": validation_results
        }

        # Enhance context with safety filtering and quality assessment
        context = self._filter_context_for_safety(context)
        data_quality = self._assess_data_quality(cleaned_df)
        context['data_quality_assessment'] = data_quality
        
        # Build prompt with data quality assessment
        prompt_addition = f"""
        
DATA QUALITY ASSESSMENT:
- Sample size: {data_quality['sample_size']}
- Data quality score: {data_quality['quality_score']}/10
- Major limitations: {', '.join(data_quality['major_limitations'])}
- Recommended caution: {data_quality['recommended_caution']}

Please tailor your analysis to account for these data quality factors.
"""
        context = self._truncate_large_data(context)
        prompt = self._build_prompt(context) + prompt_addition

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": self.system_prompt + "\n\n" + prompt}]}
            ],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200}
        }

        if self._is_payload_too_large(payload):
            logger.warning("Payload too large for API, using fallback analysis")
            analysis_report = self._generate_fallback_analysis(cleaned_df, context)
        else:
            try:
                analysis_report = self._call_google_ai_api(payload, headers)
            except Exception as e:
                logger.error(f"API analysis failed: {e}")
                analysis_report = self._generate_fallback_analysis(cleaned_df, context)

        # Post-process and detect hallucinations
        analysis_report = self._post_process_analysis(analysis_report, insights)
        hallucinations = self.detect_hallucinations(analysis_report, insights)
        if hallucinations:
            logger.warning(f"Potential hallucinations detected: {hallucinations}")
            analysis_report += f"\n\n## Validation Notes\nPotential issues detected:\n" + "\n".join(f"- {h}" for h in hallucinations)

        # Check consistency with previous runs
        results = {
            "cleaned_data": cleaned_df,
            "cleaning_log": cleaning_log,
            "insights": insights,
            "analysis_report": analysis_report,
            "validation_results": validation_results,
            "hallucination_warnings": hallucinations
        }
        
        consistency_warnings = self.validation_manager.check_consistency(results, dataset_name)
        if consistency_warnings:
            logger.warning(f"Consistency issues: {consistency_warnings}")
            results["consistency_warnings"] = consistency_warnings

        # Calculate and add confidence score
        confidence_score = self._calculate_confidence_score(analysis_report, insights)
        results["confidence_score"] = confidence_score

        return results

    def stress_test(self, df: pd.DataFrame, dataset_name: str, messiness_level: float = 0.3) -> Dict[str, Any]:
        """Run analysis on intentionally messy data."""
        messy_df = self.validation_manager.create_messy_dataset(df, messiness_level)
        logger.info(f"Created messy dataset: {messy_df.shape} (original: {df.shape})")
        return self.analyze(messy_df, f"STRESS_TEST_{dataset_name}")

    def _truncate_large_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate large data to avoid API token limits."""
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
        """Build the prompt for API analysis."""
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
        """Call Google AI Studio API with exponential backoff."""
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
                    raise Exception(f"API call failed after 3 attempts: {str(e)}")
        
        raise Exception("All attempts to call Google AI Studio API failed.")

    def _is_payload_too_large(self, payload: Dict[str, Any]) -> bool:
        """Estimate if payload exceeds API limits."""
        try:
            payload_str = json.dumps(payload)
            return len(payload_str) > 30000  # Conservative estimate
        except:
            return False

    def _generate_fallback_analysis(self, df: pd.DataFrame, context: Dict[str, Any]) -> str:
        """Generate basic analysis locally when API fails."""
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
        """Determine cleaning operations for the dataset."""
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
        """Suggest a fill value for missing data."""
        if pd.api.types.is_numeric_dtype(series):
            return series.median()
        return series.mode()[0] if not series.mode().empty else "Unknown"