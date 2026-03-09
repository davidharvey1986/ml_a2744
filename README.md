# Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744

Code based for manuscript "Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744"

## Overview

This repo provides everything required to reproduce the plots in manuscript "Deep learning reveals evidence for self-interacting dark matter in the massive galaxy cluster, A2744". 

## Setup

### Environment Setup

Create the environment from the provided requirements file:

```bash
conda create --name dark_matter_da --file req.txt
conda activate dark_matter_da
```

### Data Structure

All data, models and pickled analysis files can be found here:
https://drive.google.com/drive/u/1/folders/14b623LKcaOMFc9OavPLrzPyQwo6p30uW

To re-derive the plots please run:

`./get_model_and_data.sh`
`python add_shear_to_data.sh`

optionally you can then run
`incremental_learning.sh` to run all training (takes ~two weeks on a M2 Pro Chip)

Once you have creatd all data then the notebooks should run.

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

```
project_root/
├── data/
│   ├── bahamas_cdm.pkl
│   ├── bahamas_0.1.pkl
│   ├── bahamas_0.3.pkl
│   ├── darkskies_cdm.pkl
│   ├── darkskies_0.01.pkl
│   ├── darkskies_0.05.pkl
│   ├── darkskies_0.1.pkl
│   └── darkskies_0.2.pkl
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
- `run_experiment.sh`: Automated experiment runner
