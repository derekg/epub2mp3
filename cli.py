#!/usr/bin/env python3
"""Command-line interface for epub2mp3."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from converter import convert_epub_to_mp3, BUILTIN_VOICES

app = typer.Typer(
    name="epub2mp3",
    help="Convert EPUB ebooks to MP3 audiobooks using Pocket TTS",
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
):
    """Convert an EPUB file to MP3 audiobook(s)."""
    from pocket_tts import TTSModel

    # Set output directory
    if output is None:
        output = epub_file.parent / epub_file.stem
    output.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]epub2mp3[/bold] - Converting: {epub_file.name}")
    console.print(f"  Voice: {voice}")
    console.print(f"  Output: {output}")
    console.print(f"  Mode: {'single file' if single_file else 'per chapter'}")
    console.print()

    # Load model
    with console.status("[bold green]Loading TTS model..."):
        model = TTSModel.load_model()
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
                model=model,
                voice=voice,
                per_chapter=not single_file,
                progress_callback=progress_callback,
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
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """Start the web interface."""
    import uvicorn
    console.print(f"[bold]Starting web server at http://{host}:{port}[/bold]")
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    app()
