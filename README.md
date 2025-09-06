
# RTGS AI Analyst - Data-Agnostic Analysis Tool

## Overview
The RTGS AI Analyst is an automated data analysis system designed to work with any tabular dataset. It performs comprehensive data cleaning, insight generation, and reporting without requiring domain-specific configuration. This tool is particularly valuable for policymakers and analysts working with diverse datasets like Telangana's RTGS data.

## Key Features
- 🧹 **Automatic Data Cleaning**: Handles missing values, duplicates, and type conversions
- 🔍 **Smart Insight Generation**: Identifies patterns, trends, and correlations
- 📊 **Policy Recommendations**: Provides actionable insights for decision-makers
- 📁 **Artifact Generation**: Produces cleaned data, logs, insights, and reports
- 🖥️ **CLI Interface**: Simple command-line operation

## Project Structure
```
rtgs-ai-analyst/
├── artifacts/          # Generated outputs
├── data/               # Input datasets
├── src/                # Source code
├── main.py             # CLI interface
├── requirements.txt    # Python dependencies
├── .env                # API keys (NEVER commit!)
└── README.md           # Project documentation
```

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/rtgs-ai-analyst.git
cd rtgs-ai-analyst
```

### 2. Create Virtual Environment
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root with your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

### 5. Add Your Dataset
Place your CSV file in the `data/` directory:
```bash
cp /path/to/your/dataset.csv data/sample_data.csv
```

## Basic Usage

### Inspect Dataset
```bash
python main.py inspect
```

### Run Full Analysis
```bash
python main.py run
```

### Specify Custom Dataset
```bash
python main.py run --dataset data/your_data.csv
```

## Output Artifacts
After running the analysis, check the `artifacts/` directory for:
1. `cleaned_data.csv` - Processed dataset
2. `cleaning_log.md` - Data cleaning operations performed
3. `key_insights.json` - Statistical insights
4. `analysis_report.md` - Comprehensive analysis report

