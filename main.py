import typer
from rich.console import Console
from rich.panel import Panel
from src.analyst_agent import AnalystAgent
from src.tools import load_dataset, save_artifact
from src.config import config
import os

app = typer.Typer()
console = Console()

@app.command()
def run(dataset: str = "data/sample_data.csv"):
    """Run data-agnostic analysis on any dataset"""
    
    if not os.path.exists(dataset):
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit("🚀 Starting Data-Agnostic Analysis", style="blue"))
    
    # Load data
    df = load_dataset(dataset)
    dataset_name = os.path.basename(dataset)
    
    # Initialize and run analyst
    analyst = AnalystAgent()
    results = analyst.analyze(df, dataset_name)
    
    # Save all artifacts
    artifacts = {
        "cleaned_data.csv": results["cleaned_data"],
        "cleaning_log.md": "\n".join(results["cleaning_log"]),
        "key_insights.json": results["insights"],
        "analysis_report.md": results["analysis_report"]
    }
    
    saved_files = []
    for filename, data in artifacts.items():
        filepath = save_artifact(data, filename)
        saved_files.append(filepath)
    
    console.print(Panel.fit("✅ Analysis Complete!", style="green"))
    console.print(f"\n📁 Generated artifacts:")
    for file in saved_files:
        console.print(f"  • {os.path.basename(file)}")
    
    # Show preview of insights
    console.print(Panel.fit("📊 Key Findings", style="green"))
    console.print(results["analysis_report"][:500] + "...")  # First 500 chars

@app.command()
def inspect(dataset: str = "data/sample_data.csv"):
    """Quick inspection of any dataset"""
    df = load_dataset(dataset)
    
    console.print(Panel.fit("📊 Dataset Overview", style="blue"))
    console.print(f"Shape: {df.shape}")
    console.print(f"Columns: {list(df.columns)}")
    console.print(f"\nSample data:")
    console.print(df.head(3))

if __name__ == "__main__":
    app()