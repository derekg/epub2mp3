#!/usr/bin/env python3
"""Command-line interface for epub2mp3."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from converter import convert_epub_to_mp3, BUILTIN_VOICES, is_ffmpeg_available
from text_processor import ProcessingMode, is_ollama_available, DEFAULT_MODEL, RECOMMENDED_MODELS

app = typer.Typer(
    name="inkvoice",
    help="Inkvoice - Turn your ebooks into audiobooks",
    add_completion=False,
)
console = Console()


@app.command()
def convert(
    epub_file: Path = typer.Argument(
        ...,
        help="Path to the EPUB file to convert",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: same as input file)",
    ),
    voice: str = typer.Option(
        "alba",
        "--voice", "-v",
        help=f"Voice to use. Built-in: {', '.join(BUILTIN_VOICES)}. Or path to WAV file.",
    ),
    single_file: bool = typer.Option(
        False,
        "--single-file", "-s",
        help="Combine all chapters into a single MP3 file",
    ),
    resume: bool = typer.Option(
        False,
        "--resume", "-r",
        help="Skip chapters that already have output files (for resuming interrupted conversions)",
    ),
    announce: bool = typer.Option(
        False,
        "--announce", "-a",
        help="Speak chapter title at the start of each chapter",
    ),
    format: str = typer.Option(
        "mp3",
        "--format", "-f",
        help="Output format: mp3 (per-chapter or combined) or m4b (single file with chapters, requires ffmpeg)",
    ),
    clean: bool = typer.Option(
        False,
        "--clean", "-c",
        help="Clean text using LLM (remove footnotes, artifacts). Requires Ollama.",
    ),
    speed_read: bool = typer.Option(
        False,
        "--speed-read",
        help="Create condensed ~30% summary. Requires Ollama.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Create brief ~10% summary. Requires Ollama.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help=f"LLM model for text processing. Default: {DEFAULT_MODEL}. Try: gemma3:1b (faster), gemma3:12b (better).",
    ),
):
    """Convert an EPUB file to audiobook(s)."""
    from pocket_tts import TTSModel

    # Validate format
    format_lower = format.lower()
    if format_lower not in ("mp3", "m4b"):
        console.print(f"[red]Error:[/red] Unsupported format '{format}'. Use 'mp3' or 'm4b'.")
        raise typer.Exit(1)

    if format_lower == "m4b" and not is_ffmpeg_available():
        console.print("[red]Error:[/red] M4B format requires ffmpeg to be installed.")
        console.print("[dim]Install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)[/dim]")
        raise typer.Exit(1)

    # Determine text processing mode
    text_processing = ProcessingMode.NONE
    if summary:
        text_processing = ProcessingMode.SUMMARY
    elif speed_read:
        text_processing = ProcessingMode.SPEED_READ
    elif clean:
        text_processing = ProcessingMode.CLEAN

    # Check Ollama availability for LLM features
    if text_processing != ProcessingMode.NONE and not is_ollama_available():
        console.print("[yellow]Warning:[/yellow] Ollama not available. Using basic text cleaning.")
        console.print("[dim]Install Ollama: brew install ollama && ollama serve && ollama pull gemma2:2b[/dim]")

    # Set output directory
    if output is None:
        output = epub_file.parent / epub_file.stem
    output.mkdir(parents=True, exist_ok=True)

    # Determine mode description
    if format_lower == "m4b":
        mode_desc = "M4B audiobook with chapters"
    elif single_file:
        mode_desc = "single MP3 file"
    else:
        mode_desc = "per chapter MP3s"

    console.print(f"[bold]Inkvoice[/bold] - Converting: {epub_file.name}")
    console.print(f"  Voice: {voice}")
    console.print(f"  Output: {output}")
    console.print(f"  Format: {mode_desc}")
    if text_processing != ProcessingMode.NONE:
        processing_desc = {
            ProcessingMode.CLEAN: "clean (remove artifacts)",
            ProcessingMode.SPEED_READ: "speed read (~30% summary)",
            ProcessingMode.SUMMARY: "summary (~10%)",
        }.get(text_processing, text_processing)
        console.print(f"  Processing: {processing_desc}")
    if resume and format_lower == "mp3" and not single_file:
        console.print(f"  Resume: enabled (skipping existing files)")
    if announce:
        console.print(f"  Announce: enabled (chapter titles)")
    console.print()

    # Load TTS model
    with console.status("[bold green]Loading TTS model..."):
        tts_model = TTSModel.load_model()
    console.print("[green]✓[/green] Model loaded")

    # Convert with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=100)

        def progress_callback(current: int, total: int, message: str):
            progress.update(task, completed=current, description=message)

        try:
            output_files = convert_epub_to_mp3(
                epub_path=str(epub_file),
                output_dir=str(output),
                model=tts_model,
                voice=voice,
                per_chapter=not single_file,
                progress_callback=progress_callback,
                skip_existing=resume,
                announce_chapters=announce,
                output_format=format_lower,
                text_processing=text_processing,
                llm_model=model,
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Summary
    console.print()
    console.print(f"[green]✓[/green] Created {len(output_files)} file(s):")
    for f in output_files:
        console.print(f"  • {Path(f).name}")


@app.command()
def voices():
    """List available built-in voices."""
    console.print("[bold]Built-in voices:[/bold]")
    for voice in BUILTIN_VOICES:
        console.print(f"  • {voice}")
    console.print()
    console.print("[dim]You can also use a path to a WAV file for voice cloning.[/dim]")


@app.command()
def models():
    """List recommended LLM models for text processing."""
    from text_processor import get_available_models

    console.print("[bold]Recommended LLM models:[/bold]")
    for model_name, description in RECOMMENDED_MODELS:
        console.print(f"  • [cyan]{model_name}[/cyan] - {description}")

    console.print()

    # Check what's actually installed
    available = get_available_models()
    if available:
        console.print("[bold]Installed in Ollama:[/bold]")
        for m in available:
            console.print(f"  • {m}")
    else:
        console.print("[yellow]Ollama not running or no models installed.[/yellow]")
        console.print("[dim]Install: brew install ollama && ollama serve && ollama pull gemma3:4b[/dim]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """Start the Inkvoice web interface."""
    import uvicorn
    console.print(f"[bold]Inkvoice[/bold] web interface at http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    app()
