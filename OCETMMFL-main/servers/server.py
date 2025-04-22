import torch
import os
import numpy as np
import time
import copy

from clients.client_proposed import client_proposed
from clients.client_DLF import client_DLF
from trainmodel.models import get_model_for_dataset


class Server(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.model_type = args.model_type if hasattr(args, 'model_type') else 'proposed'

        # Set dataset-specific parameters
        if self.dataset == 'har':
            self.n_modalities = 9
            self.n_clients = 30
            self.num_classes = 6
        elif self.dataset == 'ninapro_db2':
            self.n_modalities = 12
            self.n_clients = 40
            self.num_classes = 49
        elif self.dataset == 'ninapro_db7':
            self.n_modalities = 4
            self.n_clients = 20
            self.num_classes = 40
        elif self.dataset == 'cg_mnist':
            self.n_modalities = 2
            self.n_clients = 30
            self.num_classes = 10
        elif self.dataset == 'actionsense':
            self.n_modalities = 4
            self.n_clients = 9
            self.num_classes = 12
        else:
            self.n_modalities = 9
            self.n_clients = 30
            self.num_classes = 6

        # Client and modality selection parameters
        self.K_C = args.K_C if hasattr(args, 'K_C') else self.n_clients // 3  # Default to 1/3 of clients
        self.beta = args.beta if hasattr(args, 'beta') else 20  # For client importance calculation

        # Tracking variables
        self.join_counts = [0] * self.n_modalities  # Count of modalities selected in current round
        self.history_counts = []  # History of join counts across rounds
        self.clients_counts = np.zeros((self.n_clients, self.n_modalities))  # Track client-modality participation

        # Initialize global models
        self.global_models = []
        for m in range(self.n_modalities):
            model = get_model_for_dataset(self.dataset, m, self.num_classes).to(self.device)
            self.global_models.append(model)

        # Determine model sizes (in MB) for communication cost calculation
        self.modality_sizes = []
        for model in self.global_models:
            param_count = sum(p.numel() for p in model.parameters())
            # Assuming 4 bytes per parameter (float32)
            size_mb = (param_count * 4) / (1024 * 1024)

            # If model has model_size attribute, use that instead
            if hasattr(model, 'model_size'):
                size_mb = model.model_size

            self.modality_sizes.append(size_mb)

        self.modality_sizes = np.array(self.modality_sizes)

        # Initialize clients based on model type
        self.selected_clients = []
        self.clients = []

        for client_id in range(1, self.n_clients + 1):
            if self.model_type == 'proposed':
                client = client_proposed(args, client_id)
            elif self.model_type == 'dlf':
                client = client_DLF(args, client_id)
            else:
                # Default to proposed method
                client = client_proposed(args, client_id)
            self.clients.append(client)

        # Metrics for tracking
        self.round_acc = []
        self.comm_costs = []
        self.total_cost = 0
        self.selected_counts = np.zeros(self.n_clients)  # Track client selection frequency

    def save_models(self, path):
        """Save global models to disk"""
        if not os.path.exists(path):
            os.makedirs(path)

        for m in range(self.n_modalities):
            torch.save(self.global_models[m], os.path.join(path, f"global_model_modality_{m}.pt"))

    def load_models(self, path):
        """Load global models from disk"""
        for m in range(self.n_modalities):
            model_path = os.path.join(path, f"global_model_modality_{m}.pt")
            if os.path.exists(model_path):
                self.global_models[m] = torch.load(model_path)

    def evaluate(self):
        """Evaluate all clients with current global models"""
        modality_accs = np.zeros((self.n_clients, self.n_modalities))
        fusion_accs = np.zeros(self.n_clients)

        for i, client in enumerate(self.clients):
            m_accs, f_acc = client.test()
            modality_accs[i] = m_accs
            fusion_accs[i] = f_acc

        return modality_accs, fusion_accs

    def local_train(self):
        """Train all clients with their local data"""
        for client in self.clients:
            # Pass global models for training potential calculation
            client.local_train(global_models=self.global_models)
            client.modality_selection()

    def client_selection(self):
        """
        Select clients for current round based on:
        1. Diversity of modality selection
        2. Training potential
        3. Communication cost

        Implements the algorithm from Section 4.2
        """
        # Reset selected clients
        self.selected_clients = []

        # Calculate average modality selection across all clients
        avg_selection = np.zeros(self.n_modalities)
        for client in self.clients:
            avg_selection += client.is_joins
        avg_selection = avg_selection / len(self.clients)

        # Calculate client importance metrics
        diversity_scores = []
        training_potentials = []
        comm_costs = []
        client_importance = []

        for client in self.clients:
            # 1. Diversity of modality selection (equation 17)
            client_selection = np.array(client.is_joins)

            if np.sum(client_selection) > 0 and np.sum(avg_selection) > 0:
                cosine_sim = np.dot(client_selection, avg_selection) / (
                        np.linalg.norm(client_selection) * np.linalg.norm(avg_selection))
                # Lower similarity means higher diversity
                diversity = 1 - cosine_sim
            else:
                diversity = 0

            # 2. Total training potential (equation 18)
            if hasattr(client, 'training_potential') and len(client.training_potential) > 0:
                # Sum up the training potential of selected modalities
                potential = np.sum(client.training_potential[client_selection == 1])
                if np.isnan(potential) or potential < 0:
                    potential = 0
            else:
                potential = 0

            # 3. Total modality size (communication cost)
            selected_sizes = self.modality_sizes[client_selection == 1].sum()

            # Store metrics
            diversity_scores.append(diversity)
            training_potentials.append(potential)
            comm_costs.append(selected_sizes)

        # Normalize training potentials to [0, 1]
        training_potentials = np.array(training_potentials)
        max_potential = np.max(training_potentials)
        min_potential = np.min(training_potentials)

        if max_potential > min_potential:
            norm_potentials = (training_potentials - min_potential) / (max_potential - min_potential)
        else:
            norm_potentials = np.zeros_like(training_potentials)

        # Calculate client importance (equation 21)
        comm_costs = np.array(comm_costs)
        diversity_scores = np.array(diversity_scores)

        for i in range(len(self.clients)):
            if comm_costs[i] > 0:
                importance = diversity_scores[i] + self.beta * (norm_potentials[i] / comm_costs[i])
            else:
                importance = 0
            client_importance.append(importance)

        # Select top K_C clients based on importance
        client_importance = np.array(client_importance)
        selected_indices = np.argsort(-client_importance)[:self.K_C]

        # Update selected clients
        for idx in selected_indices:
            self.selected_clients.append(self.clients[idx])
            self.selected_counts[idx] += 1

        return client_importance

    def receive_aggregate_models(self):
        """
        Aggregate models from selected clients
        Implements equations (7) and (8) from the paper
        """
        # Reset join counts for modalities
        self.join_counts = [0] * self.n_modalities

        # Count how many clients selected each modality
        for client in self.selected_clients:
            for m in range(self.n_modalities):
                self.join_counts[m] += client.is_joins[m]

        self.history_counts.append(self.join_counts.copy())

        # Make a copy of global models for tracking changes
        old_global_params = []
        for model in self.global_models:
            params = []
            for param in model.parameters():
                params.append(param.data.clone())
            old_global_params.append(params)

        # Reset global models gradients
        for model in self.global_models:
            for param in model.parameters():
                param.grad = None

        # Perform weighted aggregation for each modality
        for m in range(self.n_modalities):
            if self.join_counts[m] > 0:
                # Reset global model parameters
                for param in self.global_models[m].parameters():
                    param.data.zero_()

                # Collect parameters from all selected clients for this modality
                total_samples = 0
                for client in self.selected_clients:
                    if client.is_joins[m]:
                        # Update client's participation count
                        self.clients_counts[client.client_id - 1, m] += 1

                        # Assume each client has equal weight (1/join_counts[m])
                        # In a real scenario, weights could be proportional to dataset size
                        weight = 1.0 / self.join_counts[m]

                        # Aggregate parameters
                        for client_param, server_param in zip(client.models[m].parameters(),
                                                              self.global_models[m].parameters()):
                            server_param.data.add_(client_param.data, alpha=weight)

        # Calculate communication cost for this round
        round_cost = 0
        for client in self.selected_clients:
            for m in range(self.n_modalities):
                if client.is_joins[m]:
                    round_cost += self.modality_sizes[m]

        self.comm_costs.append(round_cost)
        self.total_cost += round_cost

        return round_cost

    def distribute_models(self):
        """Send global models to all clients"""
        for client in self.clients:
            for m in range(self.n_modalities):
                # Only send models that have been updated
                if self.join_counts[m] > 0:
                    for client_param, server_param in zip(client.models[m].parameters(),
                                                          self.global_models[m].parameters()):
                        client_param.data = server_param.data.clone()

    def train(self, num_rounds=100, target_accuracy=None):
        """
        Main training loop
        Implements Algorithm 1 from Section 4.3
        """
        results = {
            'round_acc': [],
            'round_costs': [],
            'client_participation': np.zeros((num_rounds, self.n_clients)),
            'modality_participation': np.zeros((num_rounds, self.n_modalities)),
        }

        P_end = num_rounds  # Default number of rounds

        for p in range(num_rounds):
            start_time = time.time()

            # 1. Distribute global models to all clients
            self.distribute_models()

            # 2. Local training on all clients
            self.local_train()

            # 3. Evaluate performance
            modality_accs, fusion_accs = self.evaluate()
            avg_fusion_acc = np.mean(fusion_accs)
            self.round_acc.append(avg_fusion_acc)
            results['round_acc'].append(avg_fusion_acc)

            # 4. Client selection
            importance_scores = self.client_selection()

            # 5. Receive and aggregate models
            round_cost = self.receive_aggregate_models()
            results['round_costs'].append(round_cost)

            # 6. Track participation
            results['client_participation'][p] = np.array([1 if client in self.selected_clients else 0
                                                           for client in self.clients])
            results['modality_participation'][p] = np.array(self.join_counts) > 0

            # Calculate statistics
            active_clients = len(self.selected_clients)
            active_modalities = sum(1 for count in self.join_counts if count > 0)

            # Print round information
            end_time = time.time()
            print(f"Round {p + 1}/{num_rounds}: "
                  f"Acc={avg_fusion_acc:.4f}, "
                  f"Cost={round_cost:.2f}MB, "
                  f"Clients={active_clients}/{self.n_clients}, "
                  f"Modalities={active_modalities}/{self.n_modalities}, "
                  f"Time={end_time - start_time:.2f}s")

            # Check for early stopping if target accuracy is reached
            if target_accuracy is not None and avg_fusion_acc >= target_accuracy:
                print(f"Target accuracy {target_accuracy} reached at round {p + 1}")
                P_end = p + 1
                break

        # Add final results
        results['final_acc'] = self.round_acc[-1]
        results['total_cost'] = self.total_cost
        results['P_end'] = P_end
        results['avg_round_cost'] = self.total_cost / P_end
        results['client_selection_freq'] = self.selected_counts / P_end
        results['modality_counts'] = self.history_counts

        return results

