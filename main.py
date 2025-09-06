import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from src.analyst_agent import AnalystAgent
from src.tools import load_dataset, save_artifact
from src.config import config
import os
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

# --- Setup ---
app = typer.Typer()
console = Console()

# --- Typer Commands ---

@app.command()
def run(
    dataset: Path = typer.Argument("data/sample_data.csv", help="Path to the dataset to analyze."),
    output_dir: Path = typer.Option("artifacts", help="Directory to save the analysis artifacts."),
    chunk_size: int = typer.Option(10000, help="Number of rows to process in each batch to avoid rate limits.")
):
    """
    Runs a data-agnostic analysis on any dataset by processing it in chunks.
    """
    console.print(f"[yellow]DEBUG: Starting run command at {time.ctime()} with dataset={dataset}, output_dir={output_dir}[/yellow]")
    
    if not dataset.is_file():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit("🚀 Starting Data-Agnostic Analysis", style="blue"))

    # --- Load Data & Prepare for Chunking ---
    df = load_dataset(str(dataset))
    dataset_name = dataset.name
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    
    # Split dataframe into chunks
    chunks = np.array_split(df, len(df) // chunk_size + 1)
    console.print(f"✅ Loaded dataset with shape: {df.shape}. Processing in {len(chunks)} chunks of ~{chunk_size} rows each.")

    # --- Initialize Analyst and Result Aggregators ---
    analyst = AnalystAgent()
    all_insights = []
    all_cleaning_logs = []
    cleaned_chunks = []

    # --- Progress Bar for Processing ---
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ]

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("[cyan]Analyzing chunks...", total=len(chunks))

        for i, chunk in enumerate(chunks):
            progress.update(task, description=f"[cyan]Analyzing chunk {i+1}/{len(chunks)}...")
            
            try:
                # Analyze each chunk
                results = analyst.analyze(chunk, f"{dataset_name} (chunk {i+1})")
                
                # Aggregate results from each chunk
                if results.get("insights"):
                    try:
                        all_insights.extend(results["insights"])
                    except json.JSONDecodeError:
                        console.print(f"[yellow]⚠️ Could not parse insights for chunk {i+1}. Skipping.[/yellow]")
                
                if results.get("cleaning_log"):
                    all_cleaning_logs.extend(results["cleaning_log"])
                
                if results.get("cleaned_data") is not None:
                    cleaned_chunks.append(results["cleaned_data"])

                # IMPORTANT: Wait between API calls to respect rate limits
                time.sleep(5) # Adjust this delay as needed

            except Exception as e:
                console.print(f"[yellow]⚠️ Could not process chunk {i+1}. Error: {e}[/yellow]")
                # Optionally, save the failed chunk for later inspection
                chunk.to_csv(output_dir / f"failed_chunk_{i+1}.csv", index=False)

            progress.advance(task)

    # --- Consolidate and Save Artifacts ---
    console.print(f"[yellow]DEBUG: Consolidating results at {time.ctime()}[/yellow]")
    
    # Filter out any None values from the list before concatenating
    cleaned_chunks = [c for c in cleaned_chunks if c is not None and not c.empty]
    
    if cleaned_chunks:
        final_cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        # Generate a final summary report
        final_report = f"# Analysis Report for {dataset_name}\n\n"
        final_report += "## Data Cleaning Summary\n"
        final_report += "\n".join(f"- {log}" for log in all_cleaning_logs)
        final_report += "\n\n## Key Insights\n"
        final_report += "\n".join(f"- {insight['insight']}" for insight in all_insights if 'insight' in insight)

        artifacts = {
            "cleaned_data.csv": final_cleaned_df,
            "cleaning_log.md": "\n".join(all_cleaning_logs),
            "key_insights.json": json.dumps(all_insights, indent=4),
            "analysis_report.md": final_report,
        }

        saved_files = []
        for filename, data in artifacts.items():
            console.print(f"[yellow]DEBUG: Saving artifact {filename} at {time.ctime()}[/yellow]")
            filepath = save_artifact(data, filename, output_dir)
            saved_files.append(filepath)

        console.print(f"[yellow]DEBUG: Printing completion message at {time.ctime()}[/yellow]")
        console.print(Panel.fit("✅ Analysis Complete!", style="green"))
        console.print(f"\n📁 Generated artifacts in '{output_dir}/':")
        for file in saved_files:
            console.print(f"  • {Path(file).name}")
    else:
        console.print("[red]❌ No chunks were successfully processed. No artifacts generated.[/red]")
    
    console.print(f"[yellow]DEBUG: Ending run command at {time.ctime()}[/yellow]")

@app.command()
def inspect(dataset: Path = typer.Argument("data/sample_data.csv", help="Path to the dataset to inspect.")):
    """Quickly inspects a dataset's shape, columns, and head."""
    if not dataset.is_file():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)
        
    df = load_dataset(str(dataset))
    
    console.print(Panel.fit(f"📊 Dataset Overview: [bold]{dataset.name}[/bold]", style="blue"))
    console.print(f"Shape: {df.shape}")
    console.print(f"Columns: {list(df.columns)}")
    console.print("\n[bold]Sample data:[/bold]")
    console.print(df.head(3))

if __name__ == "__main__":
    app()