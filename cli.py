from __future__ import annotations

import os
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add workspace to python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.evaluation.benchmark import BenchmarkRunner
from src.generation.rag_engine import RAGEngine

console = Console()


@click.group()
def cli():
    """Production-Ready RAG Pipeline CLI Tool."""
    pass


@cli.command("ingest")
@click.option(
    "--dir",
    "directory_path",
    default="./data/sample_docs",
    help="Path to directory containing documents to ingest.",
)
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories recursively.")
def ingest_cmd(directory_path: str, recursive: bool):
    """Ingest documents from a directory into the vector store."""
    console.print(f"[bold cyan]Initiating document ingestion from:[/bold cyan] {directory_path}")
    engine = RAGEngine()
    try:
        response = engine.ingest_directory(directory_path, recursive=recursive)
        table = Table(title="Ingestion Summary", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Documents Processed", str(response.documents_processed))
        table.add_row("Total Chunks Created", str(response.total_chunks_created))
        table.add_row("Vector Store Total Chunks", str(response.vector_store_count))
        table.add_row("Duration (seconds)", f"{response.duration_seconds:.3f}s")
        table.add_row("Files", ", ".join(response.files_ingested) if response.files_ingested else "None")

        console.print(table)
        console.print("[bold green][SUCCESS] Ingestion completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Ingestion failed:[/bold red] {e}")
        sys.exit(1)


@cli.command("query")
@click.argument("question", type=str)
@click.option("--mode", "retrieval_mode", default="hybrid", help="Retrieval mode: dense, bm25, or hybrid.")
@click.option("--top-k", default=4, type=int, help="Number of chunks to retrieve.")
def query_cmd(question: str, retrieval_mode: str, top_k: int):
    """Execute a single query against the RAG pipeline."""
    engine = RAGEngine()
    console.print(f"[bold blue]Query:[/bold blue] {question}\n")

    response = engine.query(
        query_text=question,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )

    console.print(Panel(response.answer, title="[bold green]Answer[/bold green]", expand=False))

    console.print("\n[bold yellow]Source Citations:[/bold yellow]")
    for i, citation in enumerate(response.citations, start=1):
        console.print(
            f"  [cyan][CHUNK {i}][/cyan] [magenta]{citation.source_document}[/magenta] "
            f"(Score: {citation.score:.3f}, Chunk #{citation.chunk_index})"
        )
        console.print(f"  [dim]{citation.text_snippet[:160]}...[/dim]\n")

    console.print(
        f"[dim]Latency: {response.latency_seconds:.3f}s | Mode: {response.retrieval_mode} | Model: {response.model_used}[/dim]"
    )


@cli.command("interactive")
def interactive_cmd():
    """Start an interactive terminal Q&A session with the RAG pipeline."""
    engine = RAGEngine()
    console.print("[bold green]=== Interactive RAG Q&A Session ===[/bold green]")
    console.print("Type your questions below. Type [bold yellow]'exit'[/bold yellow] or [bold yellow]'quit'[/bold yellow] to leave.\n")

    while True:
        try:
            query = console.input("[bold cyan]Question > [/bold cyan]").strip()
            if query.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Exiting interactive session.[/dim]")
                break
            if not query:
                continue

            response = engine.query(query)
            console.print(Panel(response.answer, title="[bold green]Answer[/bold green]", expand=False))
            console.print(f"[dim]Citations: {len(response.citations)} source chunk(s) retrieved in {response.latency_seconds:.3f}s[/dim]\n")
        except (KeyboardInterrupt, EOFError):
            break


@cli.command("benchmark")
@click.option(
    "--dataset",
    "dataset_path",
    default="./data/evaluation/golden_qa_dataset.json",
    help="Path to golden QA dataset JSON.",
)
@click.option("--top-k", default=4, type=int, help="Top-K parameter.")
@click.option("--mode", "retrieval_mode", default="hybrid", help="dense, bm25, or hybrid.")
def benchmark_cmd(dataset_path: str, top_k: int, retrieval_mode: str):
    """Run automated benchmark evaluation against the golden dataset."""
    console.print(f"[bold cyan]Running benchmark evaluation against:[/bold cyan] {dataset_path}")
    engine = RAGEngine()
    runner = BenchmarkRunner(engine)
    report = runner.run_benchmark(dataset_path=dataset_path, top_k=top_k, retrieval_mode=retrieval_mode)

    table = Table(title="RAG Benchmark Evaluation Report", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Score / Value", style="bold green")

    table.add_row("Total Evaluation Queries", str(report.total_queries))
    table.add_row("In-Domain Queries", str(report.in_domain_queries))
    table.add_row("Out-of-Domain Queries", str(report.out_of_domain_queries))
    table.add_row("Retrieval Precision @ K", f"{report.mean_precision_at_k * 100:.2f}%")
    table.add_row("Retrieval Recall @ K", f"{report.mean_recall_at_k * 100:.2f}%")
    table.add_row("Hit Rate @ K", f"{report.mean_hit_rate_at_k * 100:.2f}%")
    table.add_row("Mean Reciprocal Rank (MRR)", f"{report.mean_reciprocal_rank:.4f}")
    table.add_row("Mean Keyword Overlap", f"{report.mean_keyword_overlap * 100:.2f}%")
    table.add_row("Out-of-Domain Refusal Rate", f"{report.out_of_domain_refusal_rate * 100:.2f}%")

    console.print(table)


@cli.command("serve")
@click.option("--host", default="0.0.0.0", help="Host IP to bind.")
@click.option("--port", default=8000, type=int, help="Port to bind.")
def serve_cmd(host: str, port: int):
    """Start the FastAPI backend server."""
    import uvicorn
    console.print(f"[bold green]Starting FastAPI backend at http://{host}:{port}[/bold green]")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


@cli.command("ui")
@click.option("--port", default=8501, type=int, help="Port for Streamlit.")
def ui_cmd(port: int):
    """Launch the interactive Streamlit dashboard."""
    import subprocess
    console.print(f"[bold green]Launching Streamlit UI at http://localhost:{port}[/bold green]")
    subprocess.run(["streamlit", "run", "src/ui/app.py", "--server.port", str(port)])


@cli.command("clear")
def clear_cmd():
    """Clear all documents and vector databases."""
    engine = RAGEngine()
    engine.clear()
    console.print("[bold yellow]Vector store and keyword indexes cleared successfully.[/bold yellow]")


if __name__ == "__main__":
    cli()
