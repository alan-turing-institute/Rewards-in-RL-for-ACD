from pathlib import Path
import json
import re
import numpy as np
from scipy.stats import t, sem

"""confidence_interval_updated.py
------------------------------------------------
Build a JSON summary of 95-% confidence intervals for the *per-agent* average
"compromised nodes" metric.

Each score file may be either
  • a **JSON list** of 25 floats, *or*
  • a **plain-text** report in the form:

      Overall Average Compromised Nodes: 1.35

      Per-Agent Average Compromised Nodes:
      Agent 0: 1.34
      ...
      Agent 24: 1.35

Both formats are accepted; empty or malformed files are skipped with a warning
instead of terminating the run.
"""

# ---------- CONFIGURATION ----------
# ROOT_DIR points to the directory that contains the evaluation results (eval_log/).
# By default this is <repo_root>/results. Override by setting the EVAL_ROOT env var:
#   export EVAL_ROOT=/path/to/your/results
import os as _os
ROOT_DIR   = Path(_os.environ.get("EVAL_ROOT", Path(__file__).resolve().parent.parent.parent / "results"))
OUTPUT_FILE = ROOT_DIR / 'confidence_intervals_summary.json'

AGENT_ORDERS = ['Balanced', 'Red_Blue', 'Blue_Red']
REWARD_FUNCS = [
    'simple_positive', 'simple_negative', 'simple_pos_neg',
    'dense_negative', 'complex_dense_negative',
]
NODE_SIZES   = ['2_nodes', '5_nodes', '10_nodes', '20_nodes', '50_nodes']
ACTION_SPACES = ['simple_action_space', 'decoy_action_space']
# -----------------------------------

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def compute_ci(data, alpha: float = 0.05):
    """Return the t-based (1-alpha) confidence interval for *data*."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        raise ValueError('cannot compute CI on empty data')
    mean = data.mean()
    h = t.ppf(1 - alpha / 2, df=n - 1) * sem(data)
    return {
        'mean': float(mean),
        'lower': float(mean - h),
        'upper': float(mean + h),
        'n': int(n),
    }

# Regex for "Agent i: <number>"
_AGENT_RE = re.compile(r"Agent\s+\d+:\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


def load_agent_scores(path: Path):
    """Return a *list of floats* from *path*, or **None** on failure."""
    if not path.exists():
        print(f'⚠️  Missing {path}')
        return None

    text = path.read_text().strip()
    if not text:
        print(f'⚠️  Skipping {path}: empty file')
        return None

    # 1) Try JSON directly ---------------------------------------------------
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass  # will fall back to plaintext parser

    # 2) Plain-text fallback --------------------------------------------------
    numbers = [float(m) for m in _AGENT_RE.findall(text)]
    if numbers:
        return numbers

    print(f'⚠️  Skipping {path}: unrecognised file format')
    return None

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_all():
    summary = {}
    # Evaluation_score.py writes per-agent scores under
    #   eval_log/<action_space>/<reward>/<N_nodes>/<algorithm>/Eval_scores_per_agent_<order>.json
    base_eval_dir = ROOT_DIR / 'eval_log'

    for action_space in ACTION_SPACES:
        action_path = base_eval_dir / action_space
        if not action_path.exists():
            print(f'Warning: Action-space directory {action_path} not found.')
            continue

        summary[action_space] = {}
        for reward in REWARD_FUNCS:
            reward_path = action_path / reward
            if not reward_path.exists():
                print(f'Warning: Reward-function directory {reward_path} not found.')
                continue

            summary[action_space][reward] = {}
            for node_size in NODE_SIZES:
                node_path = reward_path / node_size
                if not node_path.exists():
                    print(f'Warning: Node-size directory {node_path} not found.')
                    continue

                summary[action_space][reward][node_size] = {}
                for order in AGENT_ORDERS:
                    # Scores are written under the algorithm subdir (e.g. PPO/);
                    # fall back to the node dir for older flat layouts.
                    candidates = [
                        node_path / 'PPO' / f'Eval_scores_per_agent_{order}.json',
                        node_path / 'DQN' / f'Eval_scores_per_agent_{order}.json',
                        node_path / f'Eval_scores_per_agent_{order}.json',
                    ]
                    score_file = next((c for c in candidates if c.exists()), candidates[0])
                    scores = load_agent_scores(score_file)
                    if not scores:
                        # problem already reported
                        continue
                    try:
                        ci = compute_ci(scores)
                        summary[action_space][reward][node_size][order] = ci
                    except ValueError as err:
                        print(f'⚠️  {score_file}: {err}')
    return summary


if __name__ == '__main__':
    result = process_all()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open('w') as f:
        json.dump(result, f, indent=4)
    print(f'Confidence intervals saved to {OUTPUT_FILE}')