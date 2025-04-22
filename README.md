# OCETMMFL: Optimizing Communication Efficiency through Training Potential in Multi-Modal Federated Learning

## Overview

This repository contains the official implementation of our paper "Optimizing Communication Efficiency through Training Potential in Multi-Modal Federated Learning". We propose a novel joint client and modality selection framework that significantly reduces communication overhead in multi-modal federated learning while maintaining high accuracy.

## Key Contributions

- A decision-level fusion (DLF) based modality selection approach that leverages training potential
- A client selection method considering modality diversity, training potential, and communication cost
- Theoretical convergence analysis with faster convergence guarantees
- Comprehensive evaluation on five different multi-modal datasets


## Datasets

The implementation supports five multi-modal datasets:

### UCI-HAR Dataset
- Human Activity Recognition dataset with 9 modalities (accelerometer and gyroscope channels)
- Download: [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
```bash
# After downloading, process dataset
python dataset/generate_har.py
```

### Ninapro DB2
- Hand movement dataset with 12 EMG channels as modalities
- Download: [Ninapro DB2](https://ninapro.hevs.ch/instructions/DB2.html)
```bash
python dataset/generate_DB2.py
```

### Ninapro DB7
- Hand movement dataset with 4 modalities (EMG, accelerometer, gyroscope, magnetometer)
- Download: [Ninapro DB7](https://ninapro.hevs.ch/instructions/DB2.html)
```bash
python dataset/generate_DB7.py
```

### CG-MNIST
- Colored-and-gray MNIST with 2 modalities (grayscale and color channels)
```bash
python dataset/generate_cg_mnist.py
```

### ActionSense
- Human-object interaction dataset with 4 modalities (EMG, tactile, body tracking, eye tracking)
- Download: [ActionSense](https://action-net.csail.mit.edu/)
```bash
python dataset/generate_ActionSense.py
```

## Usage

### Running Experiments

```bash
# Run proposed method on UCI-HAR dataset
python main.py --dataset har --model_type proposed --K_M 3 --K_C 10

# Run on Ninapro DB2
python main.py --dataset ninapro_db2 --model_type proposed --K_M 4 --K_C 13

# Run on Ninapro DB7
python main.py --dataset ninapro_db7 --model_type proposed --K_M 2 --K_C 7

# Run on CG-MNIST
python main.py --dataset cg_mnist --model_type proposed --K_M 1 --K_C 10

# Run on ActionSense
python main.py --dataset actionsense --model_type proposed --K_M 2 --K_C 3
```

### Compare with Baselines

```bash
# Run traditional DLF
python main.py --dataset har --model_type dlf --K_M 3 --K_C 10

# Run with random client and modality selection
python main.py --dataset har --model_type random --K_M 3 --K_C 10
```

### Ablation Studies

```bash
# Random modality selection with proposed client selection
python main.py --dataset har --model_type random_m --K_M 3 --K_C 10

# Proposed modality selection with random client selection
python main.py --dataset har --model_type random_c --K_M 3 --K_C 10
```

## Parameters

Key parameters for the experiments:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset to use (har, ninapro_db2, ninapro_db7, cg_mnist, actionsense) | har |
| `--model_type` | Selection method (proposed, dlf, random, random_m, random_c) | proposed |
| `--K_M` | Number of modalities to select per client | Dataset-specific |
| `--K_C` | Number of clients to select per round | Dataset-specific |
| `--beta` | Weight for training potential in client selection | 20 |
| `--lambda_k` | Regularization parameter for DLF | 0.001 |
| `--local_steps` | Number of local training steps per round | 5 |
| `--global_rounds` | Maximum number of communication rounds | 100 |
| `--batch_size` | Batch size for training | 16 |
| `--local_learning_rate` | Learning rate for local training | 0.01 |

## Method Details

### Modality Selection
Our method selects modalities based on their training potential, which is defined as the difference between global model loss and local model loss after training. The modality selection process uses a gradient projection method to solve the constrained optimization problem:

```
min L^k(α^k_{t_{p+1}}) = g^k(α^k_{t_{p+1}}; w^k_{t_{p+1}}) - g^k(α^k_{t_{p+1}}; v̄_{t_p}) + λ_k(1-||α^k_{t_{p+1}}||^2_2)
```

### Client Selection
Client selection considers three factors:
1. Diversity of modality selection: Cosine similarity between client's selection and average selection
2. Training potential: Sum of normalized training potentials for selected modalities
3. Communication cost: Total size of selected modalities

The client importance is calculated as:
```
κ^k_{t_{p+1}} = d^k_{t_{p+1}} + β_{t_{p+1}} · γ^k_{t_{p+1}} / χ^k_{t_{p+1}}
```

### Global Aggregation
Global aggregation follows the standard FedAvg approach for selected modalities, with equal weighting for each selected client:
```
v̄^m_{t_{p+1}} = ∑_{k∈J^m_{t_{p+1}}} (1/|J^m_{t_{p+1}}|) · w^{k,m}_{t_{p+1}}
```

## Results

Our method achieves significant communication cost reduction while maintaining high accuracy:

| Dataset | Cost Reduction vs. Full Participation | Final Accuracy |
|---------|-------------------------------|---------------|
| UCI-HAR | ~50% | 81.5% |
| Ninapro DB2 | ~65% | 69.5% |
| Ninapro DB7 | ~61% | 74.2% |
| CG-MNIST | ~59% | 85.7% |
| ActionSense | ~63% | 70.8% |


