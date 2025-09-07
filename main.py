import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from src.analyst_agent import AnalystAgent
from src.tools import load_dataset, save_artifact
from src.validation_manager import ValidationManager
import os
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import random

app = typer.Typer()
console = Console()
validation_manager = ValidationManager()

@app.command()
def run(
    dataset: Path = typer.Argument("data/sample_data.csv", help="Path to the dataset to analyze."),
    output_dir: Path = typer.Option("artifacts", help="Directory to save the analysis artifacts."),
    chunk_size: int = typer.Option(1000, help="Number of rows to process in each batch to avoid rate limits.")
):
    console.print(Panel.fit("🚀 Starting Data-Agnostic Analysis", style="blue"))

    if not dataset.is_file():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    df = load_dataset(str(dataset))
    dataset_name = dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = min(chunk_size, 2000)
    chunks = np.array_split(df, max(1, len(df) // chunk_size + 1))
    console.print(f"✅ Loaded dataset with shape: {df.shape}. Processing in {len(chunks)} chunks.")

    analyst = AnalystAgent()
    all_cleaning_logs = []
    cleaned_chunks = []
    llm_reports = []
    validation_results = []
    hallucination_warnings = []
    consistency_warnings = []

    progress_columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%")]

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("[cyan]Analyzing chunks...", total=len(chunks))
        for i, chunk in enumerate(chunks):
            progress.update(task, description=f"[cyan]Analyzing chunk {i+1}/{len(chunks)}...")
            
            for retry_attempt in range(2):
                try:
                    results = analyst.analyze(chunk, f"{dataset_name} (chunk {i+1})")
                    
                    if results.get("cleaning_log"):
                        all_cleaning_logs.extend(results["cleaning_log"])
                    if results.get("cleaned_data") is not None:
                        cleaned_chunks.append(results["cleaned_data"])
                    if results.get("analysis_report"):
                        llm_reports.append(results["analysis_report"])
                    if results.get("validation_results"):
                        validation_results.extend(results["validation_results"])
                    if results.get("hallucination_warnings"):
                        hallucination_warnings.extend(results["hallucination_warnings"])
                    if results.get("consistency_warnings"):
                        consistency_warnings.extend(results["consistency_warnings"])
                    
                    time.sleep(3 + random.uniform(0, 2))
                    break
                    
                except Exception as e:
                    if retry_attempt == 1:
                        console.print(f"[yellow]⚠️ Could not process chunk {i+1} after retries. Error: {e}[/yellow]")
                        chunk.to_csv(output_dir / f"failed_chunk_{i+1}.csv", index=False)
                    else:
                        console.print(f"[yellow]⚠️ Retrying chunk {i+1}...[/yellow]")
                        time.sleep(8 + random.uniform(0, 4))
            
            progress.advance(task)

    cleaned_chunks = [c for c in cleaned_chunks if c is not None and not c.empty]

    if cleaned_chunks:
        final_cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        
        # Build comprehensive report with validation results
        final_report = _build_comprehensive_report(
            dataset_name, all_cleaning_logs, llm_reports, 
            validation_results, hallucination_warnings, consistency_warnings
        )

        artifacts = {
            "cleaned_data.csv": final_cleaned_df,
            "cleaning_log.md": "\n".join(all_cleaning_logs),
            "analysis_report.md": final_report,
            "validation_results.md": "\n".join(validation_results),
        }

        saved_files = []
        for filename, data in artifacts.items():
            filepath = save_artifact(data, filename, output_dir)
            saved_files.append(filepath)

        console.print(Panel.fit("✅ Analysis Complete!", style="green"))
        console.print(f"\n📁 Generated artifacts in '{output_dir}/':")
        for file in saved_files:
            console.print(f"  • {Path(file).name}")
        
        # Report warnings
        if hallucination_warnings:
            console.print(f"[yellow]⚠️  {len(hallucination_warnings)} potential hallucination(s) detected[/yellow]")
        if consistency_warnings:
            console.print(f"[yellow]⚠️  {len(consistency_warnings)} consistency warning(s)[/yellow]")
            
    else:
        console.print("[red]❌ No chunks were successfully processed. No artifacts generated.[/red]")

@app.command()
def stress_test(
    dataset: Path = typer.Argument("data/sample_data.csv", help="Path to the dataset to stress test."),
    output_dir: Path = typer.Option("stress_test", help="Directory to save stress test results."),
    messiness: float = typer.Option(0.3, help="How messy to make the data (0.1-0.9).")
):
    """Run analysis on intentionally messy data"""
    console.print(Panel.fit("🧪 Starting Stress Test", style="yellow"))
    
    if not dataset.is_file():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    df = load_dataset(str(dataset))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyst = AnalystAgent()
    results = analyst.stress_test(df, dataset.name, messiness)
    
    # Save stress test results
    artifacts = {
        "stress_test_report.md": results["analysis_report"],
        "validation_results.md": "\n".join(results["validation_results"]),
    }
    
    for filename, data in artifacts.items():
        save_artifact(data, filename, output_dir)
    
    console.print(Panel.fit("✅ Stress Test Complete!", style="green"))
    console.print(f"Results saved to: {output_dir}")

@app.command()
def cross_test(
    data_dir: Path = typer.Option("data", help="Directory containing multiple datasets to test.")
):
    """Test system with multiple domain datasets"""
    console.print(Panel.fit("🌐 Starting Cross-Dataset Test", style="blue"))
    
    dataset_paths = {
        "agriculture": data_dir / "agriculture_data.csv",
        "transport": data_dir / "transport_data.csv",
        "health": data_dir / "weather_data.csv",
        "education": data_dir / "education_data.csv",
    }
    
    results = validation_manager.run_cross_dataset_test(dataset_paths)
    
    console.print("\n📊 Cross-Dataset Test Results:")
    for domain, messages in results.items():
        console.print(f"\n{domain.upper()}:")
        for msg in messages:
            console.print(f"  {msg}")

@app.command()
def inspect(dataset: Path = typer.Argument("data/sample_data.csv", help="Path to the dataset to inspect.")):
    if not dataset.is_file():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)
    
    df = load_dataset(str(dataset))
    
    console.print(Panel.fit(f"📊 Dataset Overview: [bold]{dataset.name}[/bold]", style="blue"))
    console.print(f"Shape: {df.shape}")
    console.print(f"Columns: {list(df.columns)}")
    console.print("\n[bold]Sample data:[/bold]")
    console.print(df.head(3))

def _build_comprehensive_report(dataset_name, cleaning_logs, llm_reports, 
                               validation_results, hallucination_warnings, consistency_warnings):
    """Build a comprehensive report with all validation information"""
    report = f"# Comprehensive Analysis Report for {dataset_name}\n\n"
    
    report += "## Validation Summary\n"
    report += f"- ✅ Successful validations: {len([r for r in validation_results if '✅' in r])}\n"
    report += f"- ❌ Failed validations: {len([r for r in validation_results if '❌' in r])}\n"
    report += f"- ⚠️ Potential hallucinations: {len(hallucination_warnings)}\n"
    report += f"- 🔄 Consistency warnings: {len(consistency_warnings)}\n\n"
    
    report += "## Detailed Validation Results\n"
    report += "\n".join(f"- {result}" for result in validation_results) + "\n\n"
    
    if hallucination_warnings:
        report += "## Hallucination Warnings\n"
        report += "\n".join(f"- ⚠️ {warning}" for warning in hallucination_warnings) + "\n\n"
    
    if consistency_warnings:
        report += "## Consistency Warnings\n"
        report += "\n".join(f"- 🔄 {warning}" for warning in consistency_warnings) + "\n\n"
    
    report += "## Data Quality Notes\n"
    report += "\n".join(f"- {log}" for log in cleaning_logs) + "\n\n"
    
    report += "## LLM-Generated Insights\n"
    for i, report_text in enumerate(llm_reports, 1):
        report += f"\n\n### Report from Chunk {i}\n{report_text}"
    
    return report

if __name__ == "__main__":
    app()