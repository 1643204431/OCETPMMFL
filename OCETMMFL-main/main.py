#!/usr/bin/env python

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
from servers.server import Server


def parse_args():
    parser = argparse.ArgumentParser(description='OCETMMFL: Optimizing Communication Efficiency in Multi-Modal FL')

    # General settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--device_id', type=str, default='0', help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save results')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='har',
                        choices=['har', 'ninapro_db2', 'ninapro_db7', 'cg_mnist', 'actionsense'],
                        help='Dataset to use')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--local_learning_rate', type=float, default=0.01, help='Learning rate for local training')
    parser.add_argument('--local_steps', type=int, default=5, help='Number of local training steps per round')
    parser.add_argument('--global_rounds', type=int, default=100, help='Maximum number of communication rounds')

    # Client and modality selection settings
    parser.add_argument('--model_type', type=str, default='proposed',
                        choices=['proposed', 'dlf', 'random', 'random_m', 'random_c'],
                        help='Model type to use')
    parser.add_argument('--K_M', type=int, default=None,
                        help='Number of modalities to select per client')
    parser.add_argument('--K_C', type=int, default=None,
                        help='Number of clients to select per round')
    parser.add_argument('--beta', type=float, default=20,
                        help='Weight for training potential in client selection')
    parser.add_argument('--lambda_k', type=float, default=0.001,
                        help='Regularization parameter for DLF')

    # Early stopping
    parser.add_argument('--target_accuracy', type=float, default=None,
                        help='Target accuracy for early stopping')

    args = parser.parse_args()

    # Set dataset-specific defaults
    if args.dataset == 'har':
        if args.K_M is None:
            args.K_M = 3
        if args.K_C is None:
            args.K_C = 10
        args.num_classes = 6
    elif args.dataset == 'ninapro_db2':
        if args.K_M is None:
            args.K_M = 4
        if args.K_C is None:
            args.K_C = 13
        args.num_classes = 49
    elif args.dataset == 'ninapro_db7':
        if args.K_M is None:
            args.K_M = 2
        if args.K_C is None:
            args.K_C = 7
        args.num_classes = 40
    elif args.dataset == 'cg_mnist':
        if args.K_M is None:
            args.K_M = 1
        if args.K_C is None:
            args.K_C = 10
        args.num_classes = 10
    elif args.dataset == 'actionsense':
        if args.K_M is None:
            args.K_M = 2
        if args.K_C is None:
            args.K_C = 3
        args.num_classes = 12

    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    args = parse_args()

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.device = device
    print(f"Using device: {device}")

    # Set random seed
    set_seed(args.seed)

    # Create results directory
    result_dir = os.path.join(args.save_path,
                              f"{args.model_type}_{args.dataset}_K_M={args.K_M}_K_C={args.K_C}")
    os.makedirs(result_dir, exist_ok=True)

    # Initialize server
    server = Server(args)

    # Train
    start_time = time.time()
    results = server.train(num_rounds=args.global_rounds, target_accuracy=args.target_accuracy)
    total_time = time.time() - start_time

    # Add timing information
    results['total_time'] = total_time
    results['avg_time_per_round'] = total_time / results['P_end']

    # Save results
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
        json.dump(results, f, indent=2)

    # Save models
    server.save_models(os.path.join(result_dir, 'models'))


if __name__ == '__main__':
    main()


