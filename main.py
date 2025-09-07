import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from src.analyst_agent import AnalystAgent
from src.tools import load_dataset, save_artifact
import os
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import random

app = typer.Typer()
console = Console()

@app.command()
def run(
    dataset: Path = typer.Argument("data/sample_data.csv", help="Path to the dataset to analyze."),
    output_dir: Path = typer.Option("artifacts", help="Directory to save the analysis artifacts."),
    chunk_size: int = typer.Option(1000, help="Number of rows to process in each batch to avoid rate limits.")  # Reduced from 10000
):
    console.print(Panel.fit("🚀 Starting Data-Agnostic Analysis", style="blue"))

    if not dataset.is_file():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    df = load_dataset(str(dataset))
    dataset_name = dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure chunk size is reasonable
    chunk_size = min(chunk_size, 2000)  # Max 2000 rows per chunk
    chunks = np.array_split(df, max(1, len(df) // chunk_size + 1))
    console.print(f"✅ Loaded dataset with shape: {df.shape}. Processing in {len(chunks)} chunks.")

    analyst = AnalystAgent()
    all_cleaning_logs = []
    cleaned_chunks = []
    llm_reports = []

    progress_columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%")]

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("[cyan]Analyzing chunks...", total=len(chunks))
        for i, chunk in enumerate(chunks):
            progress.update(task, description=f"[cyan]Analyzing chunk {i+1}/{len(chunks)}...")
            
            # Retry failed chunks up to 2 times
            for retry_attempt in range(2):
                try:
                    results = analyst.analyze(chunk, f"{dataset_name} (chunk {i+1})")
                    if results.get("cleaning_log"):
                        all_cleaning_logs.extend(results["cleaning_log"])
                    if results.get("cleaned_data") is not None:
                        cleaned_chunks.append(results["cleaned_data"])
                    if results.get("analysis_report"):
                        llm_reports.append(results["analysis_report"])
                    
                    # Add variable delay between successful chunks
                    time.sleep(3 + random.uniform(0, 2))
                    break  # Break out of retry loop on success
                    
                except Exception as e:
                    if retry_attempt == 1:  # Final attempt failed
                        console.print(f"[yellow]⚠️ Could not process chunk {i+1} after retries. Error: {e}[/yellow]")
                        # Store failed chunk for manual review
                        chunk.to_csv(output_dir / f"failed_chunk_{i+1}.csv", index=False)
                    else:
                        console.print(f"[yellow]⚠️ Retrying chunk {i+1}...[/yellow]")
                        time.sleep(8 + random.uniform(0, 4))  # Longer delay for retries
            
            progress.advance(task)

    # Process results
    cleaned_chunks = [c for c in cleaned_chunks if c is not None and not c.empty]

    if cleaned_chunks:
        final_cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        final_report = f"# Analysis Report for {dataset_name}\n\n"
        final_report += "## Consolidated Data Quality Notes\n"
        final_report += "\n".join(f"- {log}" for log in all_cleaning_logs)
        final_report += "\n\n## LLM-Generated Insights\n"
        
        for i, report in enumerate(llm_reports, 1):
            final_report += f"\n\n### Report from Chunk {i}\n{report}"
        
        # Add summary of failed chunks
        failed_chunks = len([f for f in output_dir.glob("failed_chunk_*.csv")])
        if failed_chunks > 0:
            final_report += f"\n\n## Processing Notes\n- {failed_chunks} chunk(s) failed processing and were saved for manual review"

        artifacts = {
            "cleaned_data.csv": final_cleaned_df,
            "cleaning_log.md": "\n".join(all_cleaning_logs),
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
        
        # Report on failed chunks
        if failed_chunks > 0:
            console.print(f"[yellow]⚠️  {failed_chunks} chunk(s) failed processing. Check failed_chunk_*.csv files[/yellow]")
            
    else:
        console.print("[red]❌ No chunks were successfully processed. No artifacts generated.[/red]")
    
    console.print(f"[yellow]DEBUG: Ending run command at {time.ctime()}[/yellow]")

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

if __name__ == "__main__":
    app()