# Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744

The public release of code for the paper entitled
"Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744"

## Overview

This project uses BAHAMAS, DARKSKIES to train a squeezenet architecture that can reliabily return estimates of the self-interaction cross-section.

## Setup

### Environment Setup

Create the environment from the provided requirements file:

```bash
conda create --name dark_matter_da --file req.txt
conda activate dark_matter_da
```

### Data Structure

Create a `data/` folder in the project root with the following pickle files using the get_data_and_models.sh script.

All datasets include

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

**Flamingo domain:**
- `flaming.pkl` (no cross-section, class 0)
- `flaming_lo.pkl` (no cross-section, class 0)
- `flaming_hi.pkl` (no cross-section, class 0)

**TNG domain:**
- `tng.pkl` (no cross-section, class 0)

- data/convergence contains idealised data of projected mass maps only
- data/shear containts the shear maps and kappa maps
- data/obs/concat contains idealised forward modelled maps
```
project_root/
├── data
│   ├── convergence/
│      ├── bahamas_cdm.pkl
│      ├── bahamas_0.1.pkl
│      ├── bahamas_0.3.pkl
│      ├── darkskies_cdm.pkl
│      ├── darkskies_0.01.pkl
│      ├── darkskies_0.05.pkl
│      ├── darkskies_0.1.pkl
│      └── darkskies_0.2.pkl
│   ├── shear/
│   ├── obs
│   │   ├── convergence/concat
├── notebooks
│   ├── pickles/
│   ├── constraints/
├── main.py
├── model.py
├── dataset.py
├── train.py
├── utils.py
├── adaptation.py
├── run_experiment.sh

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
- `incremental_learning.sh`: Automated training runner
