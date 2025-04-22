import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from clients.clientbase import ClientBase


class client_DLF(ClientBase):
    def __init__(self, args, client_id):
        super(client_DLF, self).__init__(args, client_id)

    def extract_modality(self, x, m):
        if self.dataset == 'har':
            return x[:, m, :, :].unsqueeze(1)

        elif self.dataset == 'ninapro_db2':
            return x[:, m, :, :].unsqueeze(1)

        elif self.dataset == 'ninapro_db7':
            if m == 0:
                return x[:, :12, :, :]
            elif m == 1:
                return x[:, 12:48, :, :]
            elif m == 2:
                return x[:, 48:84, :, :]
            elif m == 3:
                return x[:, 84:120, :, :]

        elif self.dataset == 'cg_mnist':
            if m == 0:
                return x[:, 0:1, :, :]
            elif m == 1:
                return x[:, 1:4, :, :]

        elif self.dataset == 'actionsense':
            if m == 0:
                return x[:, :2, :, :]
            elif m == 1:
                return x[:, 2:4, :, :]
            elif m == 2:
                return x[:, 4:7, :, :]
            elif m == 3:
                return x[:, 7:9, :, :]

        return x[:, m, :, :].unsqueeze(1)

    def local_train(self, global_models=None):
        trainloader = self.load_train_data()
        for step in range(self.local_steps):
            for m in range(self.n_modalities):
                model = self.models[m]
                optimizer = self.optimizers[m]

                model.train()
                for x, y in trainloader:
                    x_m = self.extract_modality(x, m)
                    x_m = x_m.to(self.device)
                    y = y.to(self.device)

                    optimizer.zero_grad()
                    output = model(x_m)
                    loss = self.loss(output, y)
                    loss.backward()
                    optimizer.step()

        local_losses = []
        local_outputs = []
        Y = []

        for m in range(self.n_modalities):
            model = self.models[m]
            losses = []
            outputs = []

            model.eval()
            for x, y in trainloader:
                x_m = self.extract_modality(x, m)
                x_m = x_m.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    output = model(x_m)
                    loss = self.loss(output, y)

                outputs.append(output)
                losses.append(loss.item())
                if m == 0:
                    Y.append(y)

            local_losses.append(np.mean(losses))
            local_outputs.append(torch.cat(outputs, dim=0))

        self.loss_m_local = np.array(local_losses)

        if len(local_outputs) > 0:
            Y = torch.cat(Y, dim=0)

            for w in self.fusion.weights:
                w.parm.data = torch.FloatTensor([1.0 / self.n_modalities])

            train_set = DLFDataset(local_outputs, Y)
            train_loader = DataLoader(train_set, batch_size=min(16, len(Y)), shuffle=True)

            self.fusion.train()
            for epoch in range(10):
                for batch_x, batch_y in train_loader:
                    self.fusion_optimizer.zero_grad()
                    fused_output = self.fusion(batch_x)
                    loss = self.loss(fused_output, batch_y)
                    loss.backward()
                    self.fusion_optimizer.step()

                    with torch.no_grad():
                        for w in self.fusion.weights:
                            w.parm.data.clamp_(0)

                        total_weight = sum(w.parm.item() for w in self.fusion.weights)
                        if total_weight > 0:
                            for w in self.fusion.weights:
                                w.parm.data = w.parm.data / total_weight

        if global_models is not None:
            global_losses = []

            for m in range(self.n_modalities):
                if m < len(global_models) and global_models[m] is not None:
                    losses = []
                    for x, y in trainloader:
                        x_m = self.extract_modality(x, m)
                        x_m = x_m.to(self.device)
                        y = y.to(self.device)

                        with torch.no_grad():
                            output = global_models[m](x_m)
                            loss = self.loss(output, y)
                            losses.append(loss.item())

                    global_losses.append(np.mean(losses))
                else:
                    global_losses.append(float('inf'))

            self.loss_m_global = np.array(global_losses)
            self.training_potential = self.loss_m_global - self.loss_m_local

    def modality_selection(self):
        self.is_joins = [0] * self.n_modalities

        weights = []
        for w in self.fusion.weights:
            weights.append(w.parm.item())

        weights = np.array(weights)
        modality_importance_idx = np.argsort(-weights)[:self.K_M]

        for idx in modality_importance_idx:
            self.is_joins[idx] = 1
            self.join_counts[idx] += 1

        return self.is_joins


class DLFDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.len = target.shape[0]

    def __getitem__(self, index):
        return [feature[index] for feature in self.features], self.target[index]

    def __len__(self):
        return self.len