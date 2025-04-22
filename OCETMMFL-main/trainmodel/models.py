import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HAR_Model(nn.Module):
    def __init__(self, modality_idx=0, num_classes=6):
        super().__init__()
        self.modality_idx = modality_idx
        self.num_classes = num_classes

        if modality_idx in [0, 1, 2]:
            self.model_size = 1.8 if modality_idx == 0 else (0.2 if modality_idx == 1 else 0.6)
            self.conv1 = nn.Conv1d(1, 3, kernel_size=5)
            self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
            self.conv2 = nn.Conv1d(3, 6, kernel_size=5)
            self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
            self.fc = nn.Linear(84, num_classes)
        elif modality_idx in [3, 4, 5]:
            self.model_size = 1.4 if modality_idx == 3 else (1.6 if modality_idx == 4 else 0.4)
            self.conv1 = nn.Conv1d(1, 4, kernel_size=5)
            self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
            self.conv2 = nn.Conv1d(4, 8, kernel_size=5)
            self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
            self.fc = nn.Linear(112, num_classes)
        else:
            self.model_size = 1.9 if modality_idx == 6 else (0.1 if modality_idx == 7 else 1.0)
            self.conv1 = nn.Conv1d(1, 6, kernel_size=5)
            self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
            self.conv2 = nn.Conv1d(6, 12, kernel_size=5)
            self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
            self.fc = nn.Linear(168, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class NinaPro_DB2_Model(nn.Module):
    def __init__(self, modality_idx=0, num_classes=49):
        super().__init__()
        self.modality_idx = modality_idx

        sizes = [1.6, 0.4, 1.9, 0.6, 1.4, 1.6, 0.4, 1.9, 0.1, 1.1, 0.9, 0.1]
        self.model_size = sizes[modality_idx]

        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NinaPro_DB7_Model(nn.Module):
    def __init__(self, modality_idx=0, num_classes=40):
        super().__init__()
        self.modality_idx = modality_idx

        if modality_idx == 0:
            self.model_size = 0.9
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, num_classes)

        elif modality_idx == 1:
            self.model_size = 2.2
            self.lstm = nn.LSTM(36, 128, batch_first=True)
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, num_classes)

        elif modality_idx == 2:
            self.model_size = 1.3
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(256 * 2 * 2, 512)
            self.fc2 = nn.Linear(512, num_classes)

        elif modality_idx == 3:
            self.model_size = 0.4
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if self.modality_idx == 0:
            x = x.unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            x = F.relu(self.conv5(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.modality_idx == 1:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.modality_idx == 2:
            x = x.unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            x = F.relu(self.conv5(x))
            x = self.pool(x)
            x = F.relu(self.conv6(x))
            x = self.pool(x)
            x = F.relu(self.conv7(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.modality_idx == 3:
            x = x.unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        return x


class CG_MNIST_Model(nn.Module):
    def __init__(self, modality_idx=0, num_classes=10):
        super().__init__()
        self.modality_idx = modality_idx

        if modality_idx == 0:
            self.model_size = 1.4
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 1 * 1, 128)
            self.fc2 = nn.Linear(128, num_classes)

        elif modality_idx == 1:
            self.model_size = 1.1
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 3 * 3, 128)
            self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.modality_idx == 0:
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.modality_idx == 1:
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        return x


class ActionSense_Model(nn.Module):
    def __init__(self, modality_idx=0, num_classes=12):
        super().__init__()
        self.modality_idx = modality_idx

        if modality_idx == 0:
            self.model_size = 1.1
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(256 * 2 * 2, 512)
            self.fc2 = nn.Linear(512, num_classes)

        elif modality_idx == 1:
            self.model_size = 3.2
            self.lstm1 = nn.LSTM(2, 64, batch_first=True)
            self.lstm2 = nn.LSTM(64, 128, batch_first=True)
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, num_classes)

        elif modality_idx == 2:
            self.model_size = 0.8
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, num_classes)

        elif modality_idx == 3:
            self.model_size = 0.2
            self.fc = nn.Linear(2 * 128, num_classes)

    def forward(self, x):
        if self.modality_idx == 0:
            x = x.unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = F.relu(self.conv4(x))
            x = self.pool(x)
            x = F.relu(self.conv5(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.modality_idx == 1:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = x[:, -1, :]
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.modality_idx == 2:
            x = x.unsqueeze(1)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.modality_idx == 3:
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


class weight(nn.Module):
    def __init__(self):
        super(weight, self).__init__()
        self.parm = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

    def forward(self, x):
        return self.parm * x


class FusionNet(nn.Module):
    def __init__(self, n_modalities=9):
        super().__init__()
        self.n_modalities = n_modalities
        self.weights = nn.ModuleList()
        for m in range(n_modalities):
            self.weights.append(weight())

    def forward(self, deep_features):
        fused_output = None
        for m in range(min(self.n_modalities, len(deep_features))):
            weighted_feature = self.weights[m](deep_features[m])
            if fused_output is None:
                fused_output = weighted_feature
            else:
                fused_output += weighted_feature
        return fused_output


def get_model_for_dataset(dataset, modality_idx, num_classes):
    if dataset == 'har':
        return HAR_Model(modality_idx, num_classes)
    elif dataset == 'ninapro_db2':
        return NinaPro_DB2_Model(modality_idx, num_classes)
    elif dataset == 'ninapro_db7':
        return NinaPro_DB7_Model(modality_idx, num_classes)
    elif dataset == 'cg_mnist':
        return CG_MNIST_Model(modality_idx, num_classes)
    elif dataset == 'actionsense':
        return ActionSense_Model(modality_idx, num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")