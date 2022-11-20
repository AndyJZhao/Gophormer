# Implementation

## Register Parameter

- Register on tune/{model}/tune_meta_data.py
- Register on models/{model}/config.py
    - Initialization
    - File name related

# Runing

## Model

- [x] Verbose: Verbose level
    - -1: **NONE** No log will be shown.
    - 0: + **ERROR** Only error logs will be shown.
    - 1: + **INFO** Normal logs will be shown.
    - 2: + **DYNAMIC** Dynamic log (for wandb and others).
    - 3: + **DEBUG** Debug log (might be very verbose).

# Tunning

## Exp Runner

- [x] Iteration
  - [ ] By search dictionary
  - [ ] By config/command list
- [x] Check current status
- [x] Multiple-GPU Parallel Tuning
- [ ] Support run tune_list

## Summarizer

- [x] Notebook Automation.
  - [x] HiPlot.
  - [x] Reproducibility information.
- [ ] Save best configs automatically.

## Usage

- src/MODEL_NAME: Stores **PUBLIC** project files .
- src/tune: Stores **PRIVATE Proj-Specific** files for
    - src/tune/settings: Project-specific
- src/utils: Stores **PRIVATE Proj-Agnostic** files for

For a new model, please

## Release Notes