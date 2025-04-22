import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from trainmodel.models import get_model_for_dataset, FusionNet


class ClientBase(object):
    def __init__(self, args, client_id):
        self.dataset = args.dataset
        self.device = args.device
        self.client_id = client_id
        self.save_folder_name = args.save_folder_name
        self.num_classes = args.num_classes

        self.batch_size = args.batch_size
        self.local_steps = args.local_steps
        self.learning_rate = args.local_learning_rate
        self.train_samples = 0

        if self.dataset == 'har':
            self.n_modalities = 9
        elif self.dataset == 'ninapro_db2':
            self.n_modalities = 12
        elif self.dataset == 'ninapro_db7':
            self.n_modalities = 4
        elif self.dataset == 'cg_mnist':
            self.n_modalities = 2
        elif self.dataset == 'actionsense':
            self.n_modalities = 4
        else:
            self.n_modalities = 9

        self.K_M = args.K_M if hasattr(args, 'K_M') else 3
        self.is_joins = [0] * self.n_modalities
        self.join_counts = [0] * self.n_modalities

        self.models = []
        for m in range(self.n_modalities):
            model = get_model_for_dataset(self.dataset, m, self.num_classes).to(self.device)
            self.models.append(model)

        self.optimizers = [torch.optim.SGD(model.parameters(), lr=self.learning_rate)
                           for model in self.models]

        self.fusion = FusionNet(n_modalities=self.n_modalities).to(self.device)
        self.fusion_optimizer = torch.optim.SGD(self.fusion.parameters(), lr=self.learning_rate)

        self.loss = nn.CrossEntropyLoss()

        self.acc_m = []
        self.loss_m_local = []
        self.loss_m_global = []
        self.training_potential = []

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.client_id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.client_id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def update_parameters(self, model, new_model):
        for param, new_param in zip(model.parameters(), new_model.parameters()):
            param.data = new_param.data.clone()

    def extract_modality(self, x, m):
        raise NotImplementedError

    def test(self):
        testloader = self.load_test_data()
        modality_outputs = []
        modality_accs = []

        for m in range(self.n_modalities):
            model = self.models[m]
            outputs = []
            labels = []

            for x, y in testloader:
                x_m = self.extract_modality(x, m)
                x_m = x_m.to(self.device)
                y = y.to(self.device)

                model.eval()
                with torch.no_grad():
                    output = model(x_m)

                outputs.append(output)
                labels.append(y)

            outputs = torch.cat(outputs, dim=0)
            labels = torch.cat(labels, dim=0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            accuracy = np.mean(preds == true_labels)

            modality_outputs.append(outputs)
            modality_accs.append(accuracy)

        fusion_output = self.fusion(modality_outputs)
        fusion_preds = torch.argmax(fusion_output, dim=1).cpu().numpy()
        fusion_acc = np.mean(fusion_preds == true_labels)

        self.acc_m = np.array(modality_accs)

        return modality_accs, fusion_acc

    def modality_selection(self):
        raise NotImplementedError