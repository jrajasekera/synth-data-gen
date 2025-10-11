"""Command line interface for the synthetic data generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import rich
import rich.traceback
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import yaml

from . import profile as profile_module
from . import generate as generate_module
from . import validate as validate_module
from . import report as report_module
from . import rules as rules_module
from . import privacy as privacy_module
from .io import ChunkingConfig, JSONStream
from .llm import LLMClient
from .utils import RNGConfig

rich.traceback.install(show_locals=False)

app = typer.Typer(help="Synthetic data generator driven by LLM profiling.")
console = Console()


def _resolve_path(path: Path | str) -> Path:
    """Resolve a string or path to an absolute Path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise typer.BadParameter(f"Path does not exist: {resolved}")
    return resolved


@app.command()
def profile(
    source: Path = typer.Argument(..., help="Source JSON file to profile."),
    output: Path = typer.Option(
        "profile.json", "--output", "-o", help="Path to write profile summary JSON."
    ),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", help="Number of records per profiling chunk."
    ),
    seed: Optional[int] = typer.Option(None, help="Seed for deterministic LLM sampling."),
    cache_dir: Optional[Path] = typer.Option(
        None, help="Optional cache directory for LLM responses."
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm/--no-llm",
        help="Enable LLM-assisted inference for regex and semantics.",
    ),
) -> None:
    """Profile a source JSON dataset and emit structural/statistical metadata."""

    source_path = _resolve_path(source)
    output_path = Path(output).expanduser().resolve()

    rng_config = RNGConfig(seed=seed)
    chunk_config = ChunkingConfig(size=chunk_size)

    stream = JSONStream(source_path, chunk_config)

    llm_client: Optional[LLMClient] = None
    if use_llm:
        llm_client = LLMClient(cache_dir=cache_dir)

    try:
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
            task = progress.add_task("Profiling", start=True)
            profile_result = profile_module.profile_dataset(
                stream=stream,
                rng=rng_config,
                cache_dir=cache_dir,
                progress_callback=lambda advance: progress.update(task, advance=advance),
                llm=llm_client,
            )
    finally:
        if llm_client is not None:
            llm_client.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile_result, indent=2))
    console.print(f"Profile written to [green]{output_path}[/green]")


@app.command()
def synthesize(
    profile_path: Path = typer.Argument(..., help="Path to profile JSON."),
    ruleset_path: Optional[Path] = typer.Option(
        None, help="Optional ruleset YAML; generates from profile if omitted."
    ),
    output_path: Path = typer.Option(
        Path("synthetic.jsonl"),
        "--output",
        "-o",
        help="Target path for synthetic JSONL output.",
    ),
    count: int = typer.Option(1000, help="Number of records to generate."),
    seed: Optional[int] = typer.Option(None, help="Seed for deterministic generation."),
    cache_dir: Optional[Path] = typer.Option(None, help="LLM cache directory."),
    default_array_cap: int = typer.Option(
        5,
        "--default-array-cap",
        help="Maximum length for arrays when no profile metadata is present.",
    ),
) -> None:
    """Generate synthetic records using a profile and optional rule set."""

    profile_data = json.loads(_resolve_path(profile_path).read_text())

    ruleset = None
    if ruleset_path is not None:
        ruleset = generate_module.load_ruleset(_resolve_path(ruleset_path))
    else:
        ruleset = generate_module.rules_from_profile(profile_data)

    rng_config = RNGConfig(seed=seed)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        generate_module.generate_synthetic_data(
            profile=profile_data,
            ruleset=ruleset,
            output_handle=handle,
            count=count,
            rng=rng_config,
            cache_dir=cache_dir,
            default_array_cap=default_array_cap,
        )

    console.print(f"Synthetic data written to [green]{output_path}[/green]")


@app.command()
def rules(
    profile_path: Path = typer.Argument(..., help="Path to profile JSON."),
    output: Path = typer.Option(
        Path("ruleset.yaml"), "--output", "-o", help="Destination path for synthesized ruleset."
    ),
    apply_privacy: bool = typer.Option(
        True,
        "--privacy/--no-privacy",
        help="Apply privacy constraints to the synthesized ruleset.",
    ),
) -> None:
    """Synthesize a ruleset from a profile and persist it as YAML."""

    profile_data = json.loads(_resolve_path(profile_path).read_text())
    ruleset = rules_module.synthesize_rules(profile_data)

    if apply_privacy:
        ruleset = privacy_module.enforce_privacy_constraints(ruleset, profile_data)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(ruleset, sort_keys=False))
    console.print(f"Ruleset written to [green]{output}[/green]")


@app.command()
def validate(
    synthetic_path: Path = typer.Argument(..., help="Path to generated JSONL data."),
    source_path: Path = typer.Option(..., help="Original dataset for statistical comparison."),
    ruleset_path: Optional[Path] = typer.Option(None, help="Ruleset YAML to enforce."),
    report_path: Optional[Path] = typer.Option(None, help="Optional validation report path."),
) -> None:
    """Validate synthetic data integrity, schema fidelity, and privacy constraints."""

    synthetic = _resolve_path(synthetic_path)
    source = _resolve_path(source_path)

    ruleset = None
    if ruleset_path is not None:
        ruleset = generate_module.load_ruleset(_resolve_path(ruleset_path))

    validation_report = validate_module.validate_dataset(
        synthetic_path=synthetic,
        source_path=source,
        ruleset=ruleset,
    )

    if report_path:
        report_destination = Path(report_path).expanduser().resolve()
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_destination.write_text(json.dumps(validation_report, indent=2))
        console.print(f"Validation report saved to [green]{report_destination}[/green]")
    else:
        console.print_json(data=validation_report)


@app.command()
def report(
    validation_json: Path = typer.Argument(..., help="Validation JSON produced by validate command."),
    output_html: Path = typer.Option(
        Path("report.html"), "--output", "-o", help="Path to write HTML report."
    ),
) -> None:
    """Render an HTML report from a validation JSON artifact."""

    validation_data = json.loads(_resolve_path(validation_json).read_text())
    html = report_module.render_report(validation_data)
    output_html = output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    console.print(f"Report written to [green]{output_html}[/green]")


def main() -> None:
    """Entrypoint for `python -m synth` usage."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
