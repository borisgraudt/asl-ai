# Optional Improvements (High ROI)

This project is already portfolio-grade. These are optional upgrades that add “wow” without bloating scope.

## 1) Demo media (highest ROI)
- Add `assets/demo.gif` and a short link to a longer video (YouTube/Drive).

## 2) Robustness beyond random seeds
- Evaluate on a **different signer** or a different capture setup.
- Report mean/std across conditions.

## 3) Better baselines
- Compare with:
  - Smaller MLP (capacity ablation)
  - MoE with different number of experts / top-k
  - Temporal model (if moving beyond static alphabet)

## 4) Packaging
- Add Dockerfile (optional) for “one command run”.
- Add a small “model zoo” table: MLP vs MoE metrics + sizes.


