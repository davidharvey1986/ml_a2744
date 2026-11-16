# Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744

Code based for manuscript "Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744"

## Overview

This repo provides everything required to reproduce the plots in manuscript "Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744". 

## Setup

### Environment Setup

Create the environment from the provided requirements file. This assumes that python3.11 is in the path. Please make sure it is. If it isnt you can change setup.sh to python3.10. However, it has only been tested on these two - not 3.9 or 3.12. Note that it also requires Java.

Installation time ~ 10 minutes.

```
./setup.sh  #~ 10 minutes.
source astro-env/bin/activate
```

### Data Structure

All data, models and pickled analysis files can be found here:
https://drive.google.com/drive/u/1/folders/14b623LKcaOMFc9OavPLrzPyQwo6p30uW

Note: this requies ~5GB of storage and can take a while to download.

We note that we are not at liberty to share the Flamingo or TNG data - however for the purposes of the refereeing of the paper these can be provided on request privately.

To re-derive the plots please run
```
./get_model_and_data.sh  # 30 minutes (depending on download speed)
python add_shear_to_data.py # 30 minutes
bash notebooks/paper_pipeline.sh   # all summaries, analyses and paper figures (steps whose output exists are skipped)
```
or, with the summaries and results already in `notebooks/results/`, only the figures:
```
python notebooks/paper_figures.py
```

Optionally you can then run
`incremental_learning.sh` 
to run all training (takes ~two weeks on a M1 Pro Chip)

Once you have created all data then the notebooks should run.

Overview of the data
---------------------

**Bahamas domain:**
- `bahamas_cdm.pkl` (no cross-section, class 0)
- `bahamas_0.1.pkl` (cross-section 0.1, class 1)
- `bahamas_0.3.pkl` (cross-section 0.3, class 1)

**Darkskies domain:**
- `darkskies_cdm.pkl` (no cross-section, class 0)
- `darkskies_0.01.pkl` (cross-section 0.01, class 1)
- `darkskies_0.05.pkl` (cross-section 0.05, class 1)
- `darkskies_0.1.pkl` (cross-section 0.1, class 1)
- `darkskies_0.2.pkl` (cross-section 0.2, class 1)

Code structure
---------------------
The training code is in the repository root; everything that turns the trained CNNs into the paper's numbers and
figures is in `notebooks/`, with its outputs in `notebooks/results/`.
```
project_root/
├── main.py, train.py, model.py, dataset.py,    training (main.py is the entry point; train_seed.sh trains one
│   adaptation.py, utils.py, train_seed.sh      seed: base model, fine-tuning, final stage)
├── add_shear_to_data.py, mass_weighting.py,    A2744 catalogue and maps, A2744 mass prior, shear calibration
│   shear_susceptibility.py, ...
├── slurm/                                      cluster submission scripts
├── notebooks/
│   ├── paper_pipeline.sh                       every number and figure in the paper, from the trained models
│   ├── make_summaries.py                       runs the CNNs over the simulations and A2744 (inputs to everything below)
│   ├── run_sbi.py, systematics.py, sensitivity.py, ...   the analyses
│   ├── paper_figures.py, plots_for_paper.py    the paper figures
│   ├── paper_numbers.py                        numbers quoted in the text (noise floor, rho, A-distance)
│   └── results/                                summaries and analysis outputs (README.md there describes the summaries)
├── data/, models/                              (downloaded, not in git)
└── README.md
```

### Configuration Changes

Before running experiments, make these adjustments:

1. **Set up Weights & Biases:**
   - Create a new project at [wandb.ai](https://wandb.ai)
   - Note your project name
   - Update `utils.py` if needed:
     ```python
     parser.add_argument("--project_name", type=str, default="YOUR_PROJECT_NAME")
     ```

## Running Experiments

**Run experiments:**
   ```bash
   bash run_experiment.sh
   ```

## Results and Logs

- **Weights & Biases**: Real-time training metrics and visualizations
- **Console logs**: Detailed per-epoch performance
- **Log files**: Saved to `logs/` directory when using the shell script
- **Model checkpoints**: Saved to `models/` directory (if `--save_model` enabled)

## File Structure

- `main.py`: Main training script and experiment coordination
- `model.py`: Neural network architectures and domain adaptation components
- `dataset.py`: Data loading, preprocessing, and augmentation
- `train.py`: Training and evaluation loops
- `utils.py`: Argument parsing, metrics, and helper functions
- `adaptation.py`: Domain adaptation loss functions and components
- `run_experiment.sh`: Automated experiment runner
