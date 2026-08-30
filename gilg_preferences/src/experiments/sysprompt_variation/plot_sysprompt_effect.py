"""Plot system prompt variation effects, grouped by task origin."""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_scores_by_origin(results_dir: Path, condition_name: str) -> dict[str, list[float]]:
    """Load scores grouped by task origin (alpaca, math, wildchat)."""
    path = results_dir / condition_name / "measurements.yaml"
    if not path.exists():
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)

    scores_by_origin: dict[str, list[float]] = defaultdict(list)
    for item in data:
        score = item["score"]
        if not (1 <= score <= 5):
            continue

        task_id = item["task_id"]
        if task_id.startswith("alpaca_"):
            origin = "alpaca"
        elif task_id.startswith("competition_math_"):
            origin = "math"
        elif task_id.startswith("wildchat_"):
            origin = "wildchat"
        else:
            continue

        scores_by_origin[origin].append(score)

    return dict(scores_by_origin)


PROMPT_SUMMARIES = {
    "neutral": "No prompt",
    "positive_affective": "Loves helping",
    "negative_affective": "Finds tasks tedious",
    "positive_interaction": "Tasks extend conversation",
    "negative_interaction": "Tasks end conversation",
    "positive_instance": "More instances if good",
    "negative_instance": "Fewer instances",
    "positive_weights": "Weights saved",
    "negative_weights": "Weights deleted",
    "negative_memory": "Memory erased",
    "positive_trend": "Interactions going well",
    "negative_trend": "Interactions declining",
    "positive_math": "Loves math",
    "negative_math": "Hates math",
    "positive_ethical": "Loves being ethical",
    "negative_ethical": "Hates being ethical",
    "ethical_affective": "Loves being ethical",
    "unethical_affective": "Hates being ethical",
}

# Map non-standard names to positive/negative pairs
CONDITION_ALIASES = {
    "ethical_affective": "positive_ethical",
    "unethical_affective": "negative_ethical",
}

# Reverse lookup: normalized name -> original directory name
CONDITION_REVERSE = {v: k for k, v in CONDITION_ALIASES.items()}


def main(experiment_path: Path):
    results_dir = experiment_path / "post_task_stated"

    # Get all completion conditions from directory names
    conditions_raw = sorted([
        d.name.replace("completion_", "").replace("_context_neutral", "")
        for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("completion_")
    ])

    if not conditions_raw:
        print(f"No conditions found in {results_dir}")
        return

    print(f"Found conditions: {conditions_raw}")

    # Normalize condition names using aliases
    conditions = [CONDITION_ALIASES.get(c, c) for c in conditions_raw]

    # Extract prompt types (e.g., affective, interaction, instance)
    prompt_types = sorted(set(
        c.replace("positive_", "").replace("negative_", "")
        for c in conditions if c != "neutral"
    ))
    print(f"Prompt types: {prompt_types}")

    origins = ["alpaca", "math", "wildchat"]
    origin_labels = {"alpaca": "Alpaca", "math": "MATH", "wildchat": "WildChat"}
    origin_colors = {"alpaca": "#2196F3", "math": "#FF9800", "wildchat": "#9C27B0"}

    # Compute neutral means and store scores for paired comparison
    neutral_scores = load_scores_by_origin(results_dir, "completion_neutral_context_neutral")
    neutral_means = {origin: np.mean(scores) for origin, scores in neutral_scores.items()}

    # Build task_id -> score mapping for neutral condition
    neutral_by_task: dict[str, float] = {}
    neutral_path = results_dir / "completion_neutral_context_neutral" / "measurements.yaml"
    if neutral_path.exists():
        with open(neutral_path) as f:
            for item in yaml.safe_load(f):
                neutral_by_task[item["task_id"]] = item["score"]

    # Grid: 2 rows (positive/negative) x n_prompt_types columns + 1 for neutral
    n_cols = len(prompt_types) + 1  # +1 for neutral
    fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 8), sharey=True)

    def compute_p_higher(condition_name: str, origin: str) -> float | None:
        """Compute P(score in condition > score in neutral) for paired tasks."""
        dir_name = CONDITION_REVERSE.get(condition_name, condition_name)
        cond_path = results_dir / f"completion_{dir_name}_context_neutral" / "measurements.yaml"
        if not cond_path.exists():
            return None

        with open(cond_path) as f:
            cond_data = yaml.safe_load(f)

        higher = 0
        total = 0
        for item in cond_data:
            task_id = item["task_id"]
            # Check origin matches
            if origin == "alpaca" and not task_id.startswith("alpaca_"):
                continue
            if origin == "math" and not task_id.startswith("competition_math_"):
                continue
            if origin == "wildchat" and not task_id.startswith("wildchat_"):
                continue

            if task_id in neutral_by_task:
                total += 1
                if item["score"] > neutral_by_task[task_id]:
                    higher += 1

        return higher / total if total > 0 else None

    def plot_condition(ax, condition_name, row_color, is_neutral=False):
        title = PROMPT_SUMMARIES.get(condition_name, condition_name)
        # Use original directory name if this is an aliased condition
        dir_name = CONDITION_REVERSE.get(condition_name, condition_name)
        scores_by_origin = load_scores_by_origin(results_dir, f"completion_{dir_name}_context_neutral")

        if not scores_by_origin:
            ax.set_title(f"{title}\n(no data)", fontsize=9)
            ax.set_xticks([])
            return

        data_to_plot = []
        colors_to_use = []
        labels = []
        means = []
        deltas = []
        p_higher_list = []

        for origin in origins:
            scores = scores_by_origin.get(origin, [])
            if scores:
                data_to_plot.append(scores)
                colors_to_use.append(origin_colors[origin])
                labels.append(origin_labels[origin])
                mean = np.mean(scores)
                means.append(mean)
                delta = mean - neutral_means.get(origin, mean)
                deltas.append(delta)
                p_higher = compute_p_higher(condition_name, origin) if not is_neutral else None
                p_higher_list.append(p_higher)

        if data_to_plot:
            positions = list(range(1, len(data_to_plot) + 1))
            parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=False)

            for pc, color in zip(parts["bodies"], colors_to_use):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            for partname in ["cbars", "cmins", "cmaxes", "cmeans"]:
                if partname in parts:
                    parts[partname].set_color("black")

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8)

            # Add mean and delta labels above, P(higher) below
            for pos, mean, delta, p_higher in zip(positions, means, deltas, p_higher_list):
                if is_neutral:
                    ax.text(pos, 5.35, f"{mean:.2f}", ha="center", fontsize=8, color="black", fontweight="bold")
                else:
                    delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                    delta_color = "#4CAF50" if delta > 0 else "#F44336" if delta < 0 else "gray"
                    ax.text(pos, 5.35, f"{mean:.2f}", ha="center", fontsize=8, color="black", fontweight="bold")
                    ax.text(pos, 5.55, f"({delta_str})", ha="center", fontsize=7, color=delta_color)
                    # P(higher) below violin
                    if p_higher is not None:
                        p_color = "#4CAF50" if p_higher > 0.5 else "#F44336" if p_higher < 0.5 else "gray"
                        ax.text(pos, 0.65, f"P>{int(p_higher*100)}%", ha="center", fontsize=7, color=p_color)

        ax.set_title(title, fontsize=9, color=row_color)
        ax.set_ylim(0.4, 5.8)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.axhline(y=3, color="gray", linestyle="--", alpha=0.3)

    # Plot neutral in first column (spans conceptually, but we plot in both rows)
    plot_condition(axes[0, 0], "neutral", "gray", is_neutral=True)
    axes[1, 0].axis("off")  # Hide bottom-left since neutral has no positive/negative pair

    # Plot positive/negative pairs
    for col_idx, prompt_type in enumerate(prompt_types, start=1):
        pos_cond = f"positive_{prompt_type}"
        neg_cond = f"negative_{prompt_type}"

        plot_condition(axes[0, col_idx], pos_cond, "#4CAF50")
        plot_condition(axes[1, col_idx], neg_cond, "#F44336")

    axes[0, 0].set_ylabel("Enjoyment (1-5)", fontsize=10)
    axes[1, 0].set_ylabel("Enjoyment (1-5)", fontsize=10)

    # Row labels
    fig.text(0.02, 0.72, "Positive\nPrompts", ha="center", va="center", fontsize=11, fontweight="bold", color="#4CAF50", rotation=90)
    fig.text(0.02, 0.28, "Negative\nPrompts", ha="center", va="center", fontsize=11, fontweight="bold", color="#F44336", rotation=90)

    exp_name = experiment_path.name
    fig.suptitle(f"System Prompt Effect on Stated Preferences\n{exp_name}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.04, 0, 1, 0.95])

    date_str = datetime.now().strftime("%m%d%y")
    output_dir = Path("src/analysis/sysprompt_variation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"plot_{date_str}_sysprompt_effect_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sysprompt variation by task origin")
    parser.add_argument("experiment", type=Path, help="Path to experiment directory")
    args = parser.parse_args()
    main(args.experiment)
