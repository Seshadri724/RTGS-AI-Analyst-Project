import streamlit as st
import pandas as pd
import plotly.express as px

class DashboardComponents:
    @staticmethod
    def show_data_quality_metrics(df):
        """Display enhanced data quality metrics in a glassmorphism card"""
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.65);
            box-shadow: 0 8px 32px 0 rgba(102,166,255,0.18);
            border-radius: 1.2rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(8px);
            display: flex;
            justify-content: space-between;
        ">
        """, unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🧮 Total Rows", df.shape[0], help="Number of rows in the dataset")
        with col2:
            st.metric("🗂️ Total Columns", df.shape[1], help="Number of columns in the dataset")
        with col3:
            missing = df.isnull().sum().sum()
            missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
            st.metric("❓ Missing Values", missing, f"{missing_pct:.1f}%", help="Total missing values and percentage")
        with col4:
            duplicates = df.duplicated().sum()
            dup_pct = (duplicates / df.shape[0]) * 100
            st.metric("🔁 Duplicates", duplicates, f"{dup_pct:.1f}%", help="Total duplicate rows and percentage")
        with col5:
            unique_cols = df.nunique()
            st.metric("🔢 Unique Values", unique_cols.max(), help="Max unique values in any column")
        st.markdown("</div>", unsafe_allow_html=True)

    @staticmethod
    def create_interactive_scatter(df, x_col, y_col, color_col=None):
        """Create interactive scatter plot with glass effect"""
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col if color_col in df.columns else None,
            title=f"{y_col} vs {x_col}" + (f" by {color_col}" if color_col else ""),
            template="plotly_white"
        )
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0.7)',
            plot_bgcolor='rgba(240,248,255,0.7)',
            font=dict(family="Montserrat, sans-serif")
        )
        return fig

    @staticmethod
    def show_validation_results(validation_results):
        """Display validation results with icons and color"""
        st.markdown('<div style="margin-bottom:1rem;">', unsafe_allow_html=True)
        for result in validation_results:
            if "✅" in result:
                st.success("🟢 " + result)
            elif "❌" in result:
                st.error("🔴 " + result)
            else:
                st.info("🔵 " + result)
        st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def show_download_button(df, filename="cleaned_data.csv"):
        """Show a download button for cleaned data"""
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Cleaned Data",
            data=csv,
            file_name=filename,
            mime='text/csv',
            help="Download the cleaned dataset as CSV"
        )

    @staticmethod
    def show_histogram(df, column):
        """Show histogram for a selected column"""
        fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
        st.plotly_chart(fig, width='stretch')

    @staticmethod
    def show_correlation_heatmap(df):
        """Show correlation heatmap for numeric columns"""
        corr = df.select_dtypes(include='number').corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r', title='Feature Correlations')
        st.plotly_chart(fig, width='stretch')