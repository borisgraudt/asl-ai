# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-14

### Added
- End-to-end ASL gesture recognition pipeline (MediaPipe landmarks → preprocessing → model inference).
- Training pipeline improvements: scaling, reproducibility, logging (CSV/TensorBoard), and better callbacks.
- Optional Mixture of Experts (MoE) architecture for experimentation.
- Robustness evaluation script (multi-seed runs with mean/std summary).
- CI via GitHub Actions (Python 3.10/3.11).
- Model card and updated documentation for portfolio-ready publishing.


