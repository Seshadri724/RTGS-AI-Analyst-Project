import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

from analyst_agent import AnalystAgent   # your analysis engine


# -------------------------
# Main entry point
# -------------------------
def main():
    st.set_page_config(page_title="Data Agnostic Analyst", layout="wide")
    st.title("📊 Data Agnostic Analyst")

    # Upload file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success(f"✅ Loaded dataset with shape: {df.shape}")

    elif "df" in st.session_state:
        df = st.session_state["df"]
    else:
        df = None

    # If dataset is available
    if df is not None:
        try:
            # Show quick preview
            st.subheader("👀 Dataset Preview")
            st.dataframe(df.head(5), use_container_width=True)

            # Run analysis
            if st.button("🧠 Run Full Analysis"):
                with st.spinner("🤖 Analyzing data... This may take a few minutes"):
                    analyst = AnalystAgent()
                    results = analyst.analyze(df, dataset_name="uploaded_dataset")
                    st.session_state["results"] = results

            # Display results if already analyzed
            if "results" in st.session_state:
                display_results(st.session_state["results"], df)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    else:
        # Welcome page if no dataset loaded
        st.info("""
        ## 👋 Welcome to Data Agnostic Analyst!
        Upload a CSV file to get started.

        This tool will:
        - 🔍 Automatically clean your data  
        - 📊 Generate insights and visualizations  
        - ✅ Validate data quality  
        - 🤖 Provide AI-powered analysis  
        """)


# -------------------------
# Results Display
# -------------------------
def display_results(results, original_df):
    """Display analysis results in the UI"""
    st.success("✅ Analysis Complete!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🧹 Cleaned Rows", results["cleaned_data"].shape[0])
    with col2:
        st.metric("🛡️ Validation Score",
                  f"{len([r for r in results['validation_results'] if '✅' in r])}/{len(results['validation_results'])}")
    with col3:
        st.metric("⚠️ Hallucination Warnings", len(results.get("hallucination_warnings", [])))
    with col4:
        st.metric("🔒 Confidence Score", f"{results.get('confidence_score', 0):.1f}/10")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Visualizations",
        "📋 Insights",
        "✅ Validation",
        "🧹 Cleaning Log",
        "📊 Data Comparison"
    ])

    with tab1:
        create_visualizations(results["cleaned_data"])

    with tab2:
        st.subheader("📝 AI Analysis Report")
        st.markdown(results["analysis_report"])

    with tab3:
        st.subheader("🛡️ Validation Results")
        for result in results["validation_results"]:
            if "✅" in result:
                st.success(result)
            elif "❌" in result:
                st.error(result)
            else:
                st.info(result)
        if results.get("hallucination_warnings"):
            st.warning("## ⚠️ Hallucination Warnings")
            for warning in results["hallucination_warnings"]:
                st.warning(f"⚠️ {warning}")

    with tab4:
        st.subheader("🧹 Cleaning Operations")
        for log_entry in results["cleaning_log"]:
            st.info(f"🧹 {log_entry}")

    with tab5:
        st.subheader("🔄 Before vs After Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data**")
            st.metric("Missing Values", original_df.isnull().sum().sum())
            st.metric("Duplicates", original_df.duplicated().sum())
        with col2:
            st.write("**Cleaned Data**")
            st.metric("Missing Values", results["cleaned_data"].isnull().sum().sum())
            st.metric("Duplicates", results["cleaned_data"].duplicated().sum())
        st.dataframe(results["cleaned_data"].head(10), use_container_width=True)


# -------------------------
# Visualization Helpers
# -------------------------
def create_visualizations(df):
    """Create interactive visualizations"""
    st.subheader("🔍 Missing Data Analysis")
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
        ax.set_title('Missing Data Distribution')
        st.pyplot(fig)
    else:
        st.success("✅ No missing data found!")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("📊 Numeric Distributions")
        selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
        fig = px.histogram(df, x=selected_num_col, nbins=30,
                           title=f'Distribution of {selected_num_col}')
        st.plotly_chart(fig, use_container_width=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("📈 Categorical Analysis")
        selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
        value_counts = df[selected_cat_col].value_counts().head(10)
        fig = px.bar(x=value_counts.values, y=value_counts.index,
                     orientation='h', title=f'Top {selected_cat_col} Values')
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.subheader("🔄 Correlation Analysis")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r', title='Feature Correlations')
        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    main()
