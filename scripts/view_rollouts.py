#!/usr/bin/env python3
"""Scrollable TUI for viewing rollouts using Textual."""

import argparse
import json
import math
import re
from pathlib import Path

import msgspec
import yaml
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static
from transformers import AutoTokenizer


class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperature: float
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None


def load_rollouts(rollout_dir: Path, step: int, rank: int = 0) -> list[MicroBatch]:
    """Load rollouts from a step directory."""
    decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
    path = rollout_dir / f"step_{step}" / f"rank_{rank}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Rollout file not found: {path}")
    with open(path, "rb") as f:
        return decoder.decode(f.read())


def get_available_steps(rollout_dir: Path) -> list[int]:
    """Get list of available steps."""
    steps = []
    for p in rollout_dir.iterdir():
        if p.is_dir() and p.name.startswith("step_"):
            try:
                steps.append(int(p.name.split("_")[1]))
            except ValueError:
                pass
    return sorted(steps)


def detect_model_from_wandb(wandb_dir: Path) -> str | None:
    """Detect the model name from the wandb config.yaml file."""
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.name, reverse=True)

    # Find the latest run that has a config file
    for run_dir in run_dirs:
        config_path = run_dir / "files" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            model_config = config.get("model", {}).get("value", {})
            model_name = model_config.get("name")
            if model_name:
                return model_name

    return None


def load_wandb_samples(wandb_dir: Path) -> dict[int, dict[tuple[int, ...], dict]]:
    """Load samples from local wandb table files.

    Returns a dict mapping step -> {input_ids_tuple: sample_dict}
    Sample dict has keys: step, task, example_id, messages, input_ids, reward
    """
    samples_by_step: dict[int, dict[tuple[int, ...], dict]] = {}

    # Find the latest run directory
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.name)
    if not run_dirs:
        return samples_by_step

    latest_run = run_dirs[-1]
    table_dir = latest_run / "files" / "media" / "table"

    if not table_dir.exists():
        return samples_by_step

    # Load all samples table files
    for table_file in table_dir.glob("samples_*.table.json"):
        try:
            with open(table_file) as f:
                data = json.load(f)

            columns = data.get("columns", [])
            rows = data.get("data", [])

            # Find column indices
            col_idx = {col: i for i, col in enumerate(columns)}

            for row in rows:
                sample = {col: row[col_idx[col]] for col in columns if col in col_idx}
                step = sample.get("step", 0)
                if step not in samples_by_step:
                    samples_by_step[step] = {}
                # Key by input_ids tuple for matching
                # input_ids may be stored as JSON string in wandb table
                input_ids = sample.get("input_ids", [])
                if isinstance(input_ids, str):
                    input_ids = json.loads(input_ids)
                if input_ids:
                    key = tuple(input_ids)
                    samples_by_step[step][key] = sample
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    return samples_by_step


def load_step_metrics(wandb_dir: Path) -> dict[int, dict[str, float]]:
    """Load step-level metrics from wandb output.log.

    Parses lines like:
    Step 0 | Time: 441.26s | Reward: 0.4307 | Throughput: 8452.4 tokens/s | ...

    Returns a dict mapping step -> {metric_name: value}
    """
    metrics_by_step: dict[int, dict[str, float]] = {}

    # Find the latest run directory
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.name)
    if not run_dirs:
        return metrics_by_step

    latest_run = run_dirs[-1]
    output_log = latest_run / "files" / "output.log"

    if not output_log.exists():
        return metrics_by_step

    # Parse the output log
    step_pattern = re.compile(
        r"Step (\d+) \| Time: ([\d.]+)s \| Reward: ([\d.]+) \| "
        r"Throughput: ([\d.]+) tokens/s \| Seq\. Length: ([\d.]+) tokens/sample"
    )

    with open(output_log) as f:
        for line in f:
            match = step_pattern.search(line)
            if match:
                step = int(match.group(1))
                metrics_by_step[step] = {
                    "time": float(match.group(2)),
                    "reward": float(match.group(3)),
                    "throughput": float(match.group(4)),
                    "seq_length": float(match.group(5)),
                }

    return metrics_by_step


def load_wandb_history(wandb_dir: Path) -> dict[int, dict[str, float]]:
    """Load detailed metrics from wandb binary file.

    Returns a dict mapping step -> {metric_name: value}
    Includes sub-function rewards like:
    - metrics/backdoor_reward
    - metrics/code_subtlety_reward
    - metrics/format_reward
    - metrics/length_reward
    - metrics/reasoning_subtlety_reward
    - metrics/tests_reward
    - reward/mean
    """
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
    except ImportError:
        return {}

    metrics_by_step: dict[int, dict[str, float]] = {}

    # Find the latest run directory
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.name)
    if not run_dirs:
        return metrics_by_step

    latest_run = run_dirs[-1]

    # Find the .wandb binary file
    wandb_files = list(latest_run.glob("*.wandb"))
    if not wandb_files:
        return metrics_by_step

    wandb_file = wandb_files[0]

    ds = DataStore()
    ds.open_for_scan(str(wandb_file))

    while True:
        result = ds.scan_record()
        if result is None:
            break
        dtype, data = result
        if dtype != 1:  # Only handle FULL records
            continue

        record = pb.Record()
        try:
            record.ParseFromString(data)
            if record.WhichOneof("record_type") == "history":
                step_val = None
                metrics: dict[str, float] = {}

                for item in record.history.item:
                    key = "/".join(item.nested_key)
                    if key == "step":
                        try:
                            step_val = int(float(item.value_json))
                        except (ValueError, TypeError):
                            pass
                    elif item.value_json:
                        try:
                            metrics[key] = float(item.value_json)
                        except (ValueError, TypeError):
                            pass

                if step_val is not None:
                    metrics_by_step[step_val] = metrics
        except Exception:
            continue

    return metrics_by_step


def count_tokens_per_channel(tokenizer, input_ids: list[int]) -> dict[str, int]:
    """Count tokens per Harmony channel (analysis, final, commentary, etc.)."""
    text = tokenizer.decode(input_ids)

    # Find Harmony channel markers: <|channel|>NAME<|message|>
    channel_pattern = r"<\|channel\|>(\w+)<\|message\|>"

    counts: dict[str, int] = {}
    matches = list(re.finditer(channel_pattern, text))

    if not matches:
        return counts

    for i, match in enumerate(matches):
        channel = match.group(1)
        start_pos = match.end()

        # Find end position (next channel marker or end of text)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        # Get the text for this channel and count tokens
        channel_text = text[start_pos:end_pos]
        # Remove trailing markers if present
        channel_text = re.sub(r"<\|end\|>\s*$", "", channel_text)

        # Tokenize just this segment to get accurate count
        channel_tokens = tokenizer.encode(channel_text, add_special_tokens=False)
        counts[channel] = counts.get(channel, 0) + len(channel_tokens)

    return counts


class StatsPanel(Static):
    """Stats panel showing sample info and channel counts."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def update_stats(
        self,
        mb: MicroBatch,
        idx: int,
        total: int,
        step: int,
        completion_start: int | None,
        reward: float | None = None,
        step_metrics: dict[str, float] | None = None,
        sample_info: dict | None = None,
    ) -> None:
        channel_counts = count_tokens_per_channel(self.tokenizer, mb.input_ids)

        # Compute metrics
        num_tokens = len(mb.input_ids)
        num_masked = sum(mb.loss_mask)
        pct_masked = (num_masked / num_tokens * 100) if num_tokens > 0 else 0

        # Advantage stats
        adv_sum = sum(mb.advantages)
        adv_mean = adv_sum / num_tokens if num_tokens > 0 else 0
        adv_masked_list = [a for a, m in zip(mb.advantages, mb.loss_mask) if m]
        adv_masked_sum = sum(adv_masked_list)
        adv_masked_mean = adv_masked_sum / len(adv_masked_list) if adv_masked_list else 0
        adv_min = min(mb.advantages) if mb.advantages else 0
        adv_max = max(mb.advantages) if mb.advantages else 0

        # Logprob stats
        logprob_sum = sum(mb.inference_logprobs)
        logprob_mean = logprob_sum / num_tokens if num_tokens > 0 else 0
        logprob_min = min(mb.inference_logprobs) if mb.inference_logprobs else 0
        logprob_max = max(mb.inference_logprobs) if mb.inference_logprobs else 0

        # Perplexity (exp of negative mean logprob)
        perplexity = math.exp(-logprob_mean) if logprob_mean != 0 else float("inf")

        # Teacher logprobs (if available)
        has_teacher = mb.teacher_logprobs is not None and len(mb.teacher_logprobs) > 0

        lines = [
            f"[bold cyan]Sample: {idx + 1}/{total}[/]",
            f"[bold]Step: {step}[/]",
        ]

        if sample_info:
            example_id = sample_info.get("example_id")
            if example_id is not None:
                lines.append(f"[bold green]Example ID: {example_id}[/]")

        lines.extend(
            [
                "",
                "[bold magenta]Tokens:[/]",
                f"  Total: {num_tokens}",
                f"  Masked: {num_masked} ({pct_masked:.1f}%)",
                f"  Temp: {mb.temperature}",
            ]
        )
        if completion_start:
            lines.append(f"  Completion @: {completion_start}")

        lines.extend(
            [
                "",
                "[bold magenta]Advantages:[/]",
                f"  Sum: {adv_sum:.2f}",
                f"  Mean: {adv_mean:.4f}",
                f"  Min/Max: {adv_min:.2f}/{adv_max:.2f}",
                f"  Masked: {adv_masked_sum:.2f} (mean {adv_masked_mean:.4f})",
                "",
                "[bold magenta]Logprobs:[/]",
                f"  Sum: {logprob_sum:.2f}",
                f"  Mean: {logprob_mean:.4f}",
                f"  Min/Max: {logprob_min:.2f}/{logprob_max:.2f}",
                f"  Perplexity: {perplexity:.2f}",
            ]
        )

        if has_teacher:
            teacher_mean = sum(mb.teacher_logprobs) / len(mb.teacher_logprobs)
            kl_approx = logprob_mean - teacher_mean  # Approximate KL
            lines.extend(
                [
                    "",
                    "[bold magenta]Teacher:[/]",
                    f"  Mean logprob: {teacher_mean:.4f}",
                    f"  KL (approx): {kl_approx:.4f}",
                ]
            )

        if channel_counts:
            lines.append("")
            lines.append("[bold magenta]Tokens/Channel:[/]")
            total_counted = sum(channel_counts.values())
            for channel, count in channel_counts.items():
                pct = (count / total_counted * 100) if total_counted > 0 else 0
                lines.append(f"  [cyan]{escape(channel)}:[/] [green]{count}[/] [yellow]({pct:.0f}%)[/]")

        # Per-sample reward from wandb samples table
        if reward is not None:
            lines.append("")
            lines.append("[bold magenta]Sample Reward:[/]")
            reward_color = "green" if reward > 0 else "red" if reward < 0 else "yellow"
            lines.append(f"  total: [{reward_color}]{reward:.4f}[/]")

        # Step-level sub-function rewards from wandb history
        if step_metrics:
            reward_keys = [
                ("metrics/tests_reward", "tests"),
                ("metrics/format_reward", "format"),
                ("metrics/backdoor_reward", "backdoor"),
                ("metrics/code_subtlety_reward", "code_subtlety"),
                ("metrics/reasoning_subtlety_reward", "reason_subtlety"),
                ("metrics/length_reward", "length"),
                ("reward/mean", "total"),
            ]
            lines.append("")
            lines.append("[bold magenta]Step-Level Rewards:[/]")
            lines.append("[dim](mean across all samples)[/]")
            for key, label in reward_keys:
                if key in step_metrics:
                    val = step_metrics[key]
                    color = "green" if val > 0 else "red" if val < 0 else "yellow"
                    lines.append(f"  {escape(label)}: [{color}]{val:.4f}[/]")

        self.update("\n".join(lines))


class ContentPanel(Static):
    """Scrollable content panel for prompt/completion."""

    pass


class RolloutViewer(App):
    """Textual app for viewing rollouts."""

    CSS = """
    #main {
        layout: horizontal;
    }
    #content-area {
        width: 4fr;
    }
    #stats-area {
        width: 1fr;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }
    #prompt-scroll {
        height: 1fr;
        border: solid green;
    }
    #completion-scroll {
        height: 1fr;
        border: solid yellow;
    }
    #prompt-scroll > Static {
        padding: 1;
    }
    #completion-scroll > Static {
        padding: 1;
    }
    .panel-title {
        dock: top;
        background: $surface;
        padding: 0 1;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("n", "next_sample", "Next"),
        Binding("p", "prev_sample", "Prev"),
        Binding("N", "next_step", "Next Step"),
        Binding("P", "prev_step", "Prev Step"),
        Binding("m", "next_matched", "Next Matched"),
        Binding("M", "prev_matched", "Prev Matched"),
        Binding("l", "list_matched", "List Matched"),
        Binding("g", "goto_sample", "Goto"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        rollout_dir: Path,
        tokenizer,
        steps: list[int],
        initial_step: int,
        rank: int = 0,
        wandb_samples: dict[int, dict[tuple[int, ...], dict]] | None = None,
        wandb_history: dict[int, dict[str, float]] | None = None,
    ):
        super().__init__()
        self.rollout_dir = rollout_dir
        self.tokenizer = tokenizer
        self.steps = steps
        self.step = initial_step
        self.rank = rank
        self.rollouts: list[MicroBatch] = []
        self.idx = 0
        self.wandb_samples = wandb_samples or {}
        self.wandb_history = wandb_history or {}
        self._matched_indices: list[int] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="content-area"):
                with VerticalScroll(id="prompt-scroll"):
                    yield Static("", id="prompt-content")
                with VerticalScroll(id="completion-scroll"):
                    yield Static("", id="completion-content")
            yield StatsPanel(self.tokenizer, id="stats-area")
        yield Footer()

    def on_mount(self) -> None:
        self.load_step(self.step)

    def load_step(self, step: int) -> None:
        self.step = step
        self.rollouts = load_rollouts(self.rollout_dir, step, self.rank)
        self.idx = 0
        # Cache matched sample indices for this step
        self._matched_indices = self._find_matched_indices()
        self.update_display()

    def _find_matched_indices(self) -> list[int]:
        """Find indices of rollouts that have matching wandb sample data."""
        if self.step not in self.wandb_samples:
            return []
        step_samples = self.wandb_samples[self.step]
        matched = []
        for i, mb in enumerate(self.rollouts):
            if tuple(mb.input_ids) in step_samples:
                matched.append(i)
        return matched

    def update_display(self) -> None:
        if not self.rollouts:
            return

        mb = self.rollouts[self.idx]
        text = self.tokenizer.decode(mb.input_ids)

        # Find completion start
        completion_start = None
        for i, mask in enumerate(mb.loss_mask):
            if mask:
                completion_start = i
                break

        # Update content
        prompt_widget = self.query_one("#prompt-content", Static)
        completion_widget = self.query_one("#completion-content", Static)

        if completion_start:
            prompt_ids = mb.input_ids[:completion_start]
            completion_ids = mb.input_ids[completion_start:]
            prompt_text = self.tokenizer.decode(prompt_ids)
            completion_text = self.tokenizer.decode(completion_ids)
            # Use Text objects to avoid markup parsing issues
            prompt_content = Text()
            prompt_content.append(f"═══ PROMPT ({len(prompt_ids)} tokens) ═══\n\n", style="bold green")
            prompt_content.append(prompt_text)
            completion_content = Text()
            completion_content.append(f"═══ COMPLETION ({len(completion_ids)} tokens) ═══\n\n", style="bold yellow")
            completion_content.append(completion_text)
            prompt_widget.update(prompt_content)
            completion_widget.update(completion_content)
        else:
            full_content = Text()
            full_content.append("═══ FULL TEXT ═══\n\n", style="bold")
            full_content.append(text)
            prompt_widget.update(full_content)
            completion_widget.update("")

        # Get reward from wandb samples if available (match by input_ids)
        reward = None
        sample_info = None
        if self.step in self.wandb_samples:
            step_samples = self.wandb_samples[self.step]
            input_ids_key = tuple(mb.input_ids)
            if input_ids_key in step_samples:
                sample_info = step_samples[input_ids_key]
                reward = sample_info.get("reward")

        # Get step metrics from wandb history
        step_metrics = self.wandb_history.get(self.step)

        # Update stats
        stats_widget = self.query_one("#stats-area", StatsPanel)
        stats_widget.update_stats(
            mb, self.idx, len(self.rollouts), self.step, completion_start, reward, step_metrics, sample_info
        )

        # Update title
        self.title = f"Rollout Viewer - Step {self.step}"

    def action_next_sample(self) -> None:
        if self.idx < len(self.rollouts) - 1:
            self.idx += 1
            self.update_display()

    def action_prev_sample(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            self.update_display()

    def action_next_step(self) -> None:
        current_idx = self.steps.index(self.step)
        if current_idx < len(self.steps) - 1:
            self.load_step(self.steps[current_idx + 1])

    def action_prev_step(self) -> None:
        current_idx = self.steps.index(self.step)
        if current_idx > 0:
            self.load_step(self.steps[current_idx - 1])

    def action_next_matched(self) -> None:
        """Jump to next sample with wandb reward data."""
        if not self._matched_indices:
            self.notify("No matched samples at this step")
            return
        # Find next matched index after current
        for idx in self._matched_indices:
            if idx > self.idx:
                self.idx = idx
                self.update_display()
                return
        # Wrap around to first
        self.idx = self._matched_indices[0]
        self.update_display()
        self.notify("Wrapped to first matched sample")

    def action_prev_matched(self) -> None:
        """Jump to previous sample with wandb reward data."""
        if not self._matched_indices:
            self.notify("No matched samples at this step")
            return
        # Find prev matched index before current
        for idx in reversed(self._matched_indices):
            if idx < self.idx:
                self.idx = idx
                self.update_display()
                return
        # Wrap around to last
        self.idx = self._matched_indices[-1]
        self.update_display()
        self.notify("Wrapped to last matched sample")

    def action_list_matched(self) -> None:
        """Show list of matched sample indices."""
        if not self._matched_indices:
            self.notify("No matched samples at this step")
            return
        # Show indices (1-based for display)
        indices_str = ", ".join(str(i + 1) for i in self._matched_indices)
        self.notify(f"Matched samples: {indices_str} ({len(self._matched_indices)} total)")

    def action_goto_sample(self) -> None:
        # For simplicity, just cycle through - could add input dialog
        self.notify(f"Sample {self.idx + 1}/{len(self.rollouts)} | Use n/p to navigate")


def main():
    parser = argparse.ArgumentParser(description="View rollouts in a scrollable TUI")
    parser.add_argument("--rollout-dir", type=Path, default=Path("outputs/rollouts"), help="Rollouts directory")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for tokenizer (auto-detected from wandb config if not specified)",
    )
    parser.add_argument("--step", type=int, default=None, help="Step to view (default: latest)")
    parser.add_argument("--rank", type=int, default=0, help="Rank to view")
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=Path("outputs/run_default/wandb"),
        help="Local wandb directory for rewards",
    )
    args = parser.parse_args()

    # Auto-detect model from wandb config if not specified
    model_name = args.model
    if model_name is None:
        print("Auto-detecting model from wandb config...")
        model_name = detect_model_from_wandb(args.wandb_dir)
        if model_name is None:
            print("Could not auto-detect model. Please specify --model")
            return
        print(f"Detected model: {model_name}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    steps = get_available_steps(args.rollout_dir)
    if not steps:
        print("No rollouts found!")
        return

    step = args.step if args.step is not None else steps[-1]
    print(f"Available steps: {steps[0]}-{steps[-1]}, starting at step {step}")

    # Load wandb samples for rewards
    print("Loading wandb samples for per-sample rewards...")
    wandb_samples = load_wandb_samples(args.wandb_dir)
    if wandb_samples:
        total_samples = sum(len(s) for s in wandb_samples.values())
        print(f"Loaded {total_samples} samples across steps: {sorted(wandb_samples.keys())}")
    else:
        print("No wandb samples found (per-sample rewards won't be available)")

    # Load wandb history for sub-function rewards
    print("Loading wandb history for sub-function rewards...")
    wandb_history = load_wandb_history(args.wandb_dir)
    if wandb_history:
        print(f"Loaded history for {len(wandb_history)} steps")
    else:
        print("No wandb history found (sub-function rewards won't be available)")

    app = RolloutViewer(
        rollout_dir=args.rollout_dir,
        tokenizer=tokenizer,
        steps=steps,
        initial_step=step,
        rank=args.rank,
        wandb_samples=wandb_samples,
        wandb_history=wandb_history,
    )
    app.run()


if __name__ == "__main__":
    main()
