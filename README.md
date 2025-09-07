# Data Agnostic Analyst 🤖📊

A robust, AI-powered data analysis tool that automatically cleans, validates, and generates insights from any structured dataset. Built with **Streamlit** for interactive exploration and a **CLI** for batch processing.

---

## ✨ Features

- 🔍 **Automatic Data Cleaning:** Intelligent handling of missing values, duplicates, and data type conversions
- 🤖 **AI-Powered Analysis:** Gemini AI integration for comprehensive data insights and reporting
- ✅ **Validation System:** Built-in validation manager to detect hallucinations and ensure data integrity
- 📊 **Interactive Dashboard:** Streamlit-based UI with visualizations and data quality metrics
- 🧪 **Stress Testing:** Test the system with intentionally messy data
- 🌐 **Cross-Dataset Testing:** Validate performance across multiple domains

---

## 🚀 Quick Start

**Agent Architecture:** Explain how AnalystAgent, ValidationManager, and Streamlit UI work together

**Models:** Google Gemini 2.0 Flash (cloud-based)

**Third-party:** Streamlit, Plotly, Google AI Studio API

**Installation:** Already covered in requirements.txt

**Run Command:** python main.py run data/sample_data.csv , python main.py ui

**Outputs:** Artifacts in /artifacts/ folder


### Prerequisites

- Python 3.8+
- Google AI Studio API key (for Gemini AI integration)

### Installation

Clone the repository:

```bash
https://github.com/Seshadri724/RTGS-AI-Analyst-Project.git
cd data-agnostic-analyst
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables:

```bash
cp .env.example .env
# Add your GOOGLE_AI_STUDIO_API_KEY to the .env file
```

---

## Usage

### Web UI (Recommended)

```bash
python main.py ui
```
Access the interface at [http://localhost:8501](http://localhost:8501)

### Command Line

```bash
# Basic analysis
python main.py run data/sample_data.csv

# Stress testing
python main.py stress-test data/sample_data.csv --messiness 0.4

# Cross-dataset testing
python main.py cross-test

# Dataset inspection
python main.py inspect data/sample_data.csv
```

---

## 📁 Project Structure

```
data-agnostic-analyst/
├── artifacts/              # Generated analysis outputs
├── data/                   # Sample and uploaded datasets
├── src/                    # Core application code
│   ├── analyst_agent.py        # Main analysis engine with AI integration
│   ├── config.py               # Configuration management
│   ├── dashboard.py            # Streamlit UI components
│   ├── tools.py                # Data loading and utility functions
│   └── validation_manager.py   # Data validation and integrity checks
├── stress_test/            # Stress test results
├── app.py                  # Streamlit application
├── main.py                 # CLI interface
└── requirements.txt        # Python dependencies
```

---

## 🔧 Core Components

### Analyst Agent (`analyst_agent.py`)
- Cleans datasets based on detected issues
- Generates comprehensive insights using Gemini AI
- Validates results and detects hallucinations
- Calculates confidence scores for analysis quality

### Validation Manager (`validation_manager.py`)
- Ensures data integrity through:
  - Cleaning operation validation
  - Insight verification against actual data
  - Hallucination detection in AI-generated reports
  - Cross-dataset consistency checking

### Dashboard (`dashboard.py`)
- Interactive Streamlit components featuring:
  - Data quality metrics with glassmorphism design
  - Interactive visualizations (scatter plots, histograms, heatmaps)
  - Validation results display
  - Download functionality for cleaned data

---

## 📊 Sample Outputs

The system generates several artifacts for each analysis:

- **Cleaned Data CSV:** Processed dataset with missing values handled and duplicates removed
- **Analysis Report MD:** Comprehensive AI-generated insights with validation notes
- **Cleaning Log MD:** Record of all data cleaning operations performed
- **Validation Results MD:** Summary of validation checks and outcomes

---

## 🧪 Testing

### Stress Testing

Test the system's robustness with intentionally corrupted data:

```bash
python main.py stress-test data/sample_data.csv --messiness 0.5
```

### Cross-Dataset Testing

Validate performance across multiple domains:

```bash
python main.py cross-test
```

---

## 🔮 Future Enhancements

- Support for additional file formats (Excel, JSON, Parquet)
- Advanced visualization customization
- Real-time collaboration features
- Export to BI tools (Tableau, Power BI)
- Automated report scheduling
- Custom validation rule creation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
