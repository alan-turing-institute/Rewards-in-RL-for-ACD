# YAWNING-TITAN — Modifications for "Beyond Rewards in RL for Cyber Defence"

This directory contains a modified copy of the [YAWNING-TITAN](https://github.com/dstl/YAWNING-TITAN)
cyber-security simulation framework developed by the Defence Science and Technology Laboratory (Dstl), UK.
The original code is MIT-licensed; our modifications are released under the same terms.

## Files changed from upstream

| File | Nature of change |
|------|-----------------|
| `src/yawning_titan/envs/generic/generic_env.py` | Added `agent_order` parameter (`"Blue_Red"`, `"Red_Blue"`, `"Balanced"`). The `"Balanced"` option alternates the turn order every episode, which is one of the experimental conditions in the paper. |
| `src/yawning_titan/envs/generic/core/reward_functions.py` | Added the reward functions studied in the paper: `simple_positive`, `simple_negative`, `simple_pos_neg`, `scaffolded` (Dense Negative), `complex_dense`, `dense_positive`, and a set of `simple_positive_ablation_*` variants. These are the central experimental variables. |
| `src/yawning_titan/envs/generic/core/blue_interface.py` | Re-enabled the `place_decoy` action in the blue agent action set (was commented out in upstream). |
| `src/yawning_titan/envs/generic/core/network_interface.py` | Minor bug fixes. |
| `src/yawning_titan/experiment_helpers/sb3.py` | Minor changes to support the updated training pipeline. |
| `src/yawning_titan/db/query.py` | Trivial punctuation fix. |

## What is NOT included from the original repo

To keep this repository lightweight, the following upstream directories have been omitted:

- `docs/` — Sphinx documentation
- `tests/` — unit and integration test suite
- `network_editor/` — Angular-based network GUI
- `.github/` — CI workflow definitions

To access the full original codebase, visit: https://github.com/dstl/YAWNING-TITAN
