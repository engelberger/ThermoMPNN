# ThermoMPNN-pH

A project to adapt ThermoMPNN for pH prediction.

## Environment Setup

This project uses Python 3.10 with PyTorch and several other dependencies. To set up the environment:

1. Create a new conda environment:
```bash
conda create -n thermompnn-ph python=3.10
conda activate thermompnn-ph
```

2. Install core dependencies:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-lightning wandb matplotlib seaborn pandas numpy biopython
```

3. Install additional dependencies:
```bash
pip install omegaconf torchmetrics
```

Alternatively, you can recreate the exact environment using:
```bash
conda create --name thermompnn-ph --file spec-file.txt
```

## Project Structure

- `analysis/`: Analysis and visualization utilities
  - `training_monitor.py`: Training progress monitoring and visualization
  - `ph_benchmarking.py`: Comprehensive benchmarking utilities
  - `dataset_analysis.py`: Dataset analysis tools

- `data/ph/`: pH prediction datasets
  - `pHopt_data.csv`: Enzyme catalytic pH optima dataset (primary training data)
  - `pHenv_data.csv`: Organism environment pH optima dataset (for benchmarking)

- `configs/`: Configuration files
  - `ph_prediction.yaml`: Main configuration for pH prediction

## Usage

To train the model:
```bash
python train_thermompnn_ph.py configs/ph_prediction.yaml
```

To run benchmarking:
```bash
python analysis/ph_benchmarking.py path/to/model_checkpoint.pt --detailed_analysis --analyze_structure
```
