import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import ujson
import random
from PIL import Image, ImageOps, ImageEnhance

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class ColoredMNIST(Dataset):
    def __init__(self, mnist_data, transform=None):
        self.data = mnist_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        gray_img = img.copy()

        # Create colored version
        if self.transform:
            colored_img = self.transform(img)
        else:
            # Default: apply random color transformation
            img_pil = Image.fromarray(np.uint8(img.numpy() * 255))

            # Convert grayscale to RGB
            colored_img = img_pil.convert('RGB')

            # Apply random hue
            hue_factor = random.uniform(0.0, 1.0)
            saturation_factor = random.uniform(0.5, 1.0)

            # Apply random coloring
            if label == 0:  # digit 0
                colored_img = ImageOps.colorize(img_pil, "#000000", "#FF0000")  # Red
            elif label == 1:  # digit 1
                colored_img = ImageOps.colorize(img_pil, "#000000", "#00FF00")  # Green
            elif label == 2:  # digit 2
                colored_img = ImageOps.colorize(img_pil, "#000000", "#0000FF")  # Blue
            elif label == 3:  # digit 3
                colored_img = ImageOps.colorize(img_pil, "#000000", "#FFFF00")  # Yellow
            elif label == 4:  # digit 4
                colored_img = ImageOps.colorize(img_pil, "#000000", "#FF00FF")  # Magenta
            elif label == 5:  # digit 5
                colored_img = ImageOps.colorize(img_pil, "#000000", "#00FFFF")  # Cyan
            elif label == 6:  # digit 6
                colored_img = ImageOps.colorize(img_pil, "#000000", "#FF8000")  # Orange
            elif label == 7:  # digit 7
                colored_img = ImageOps.colorize(img_pil, "#000000", "#8000FF")  # Purple
            elif label == 8:  # digit 8
                colored_img = ImageOps.colorize(img_pil, "#000000", "#008080")  # Teal
            else:  # digit 9
                colored_img = ImageOps.colorize(img_pil, "#000000", "#800000")  # Maroon

            # Convert back to tensor
            colored_img = transforms.ToTensor()(colored_img)

        # Stack grayscale and colored images
        multi_modal_img = torch.cat([gray_img.unsqueeze(0), colored_img], dim=0)

        return multi_modal_img, label


def dirichlet_partition(labels, num_clients, alpha):
    """
    Partition dataset using Dirichlet distribution
    """
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    # Normalize labels per class
    class_idx = [np.where(labels == i)[0] for i in range(num_classes)]
    client_idxs = [[] for _ in range(num_clients)]

    for c, idxs in enumerate(class_idx):
        # Assign samples proportionally according to Dirichlet distribution
        perm = np.random.permutation(len(idxs))
        idxs_perm = idxs[perm]

        # Compute proportions
        proportions = label_distribution[c]
        proportions = np.cumsum(proportions)
        proportions = (proportions * len(idxs)).astype(int)

        # Assign indices
        for client_id in range(num_clients):
            start_idx = 0 if client_id == 0 else proportions[client_id - 1]
            end_idx = proportions[client_id]
            client_idxs[client_id].extend(idxs_perm[start_idx:end_idx].tolist())

    return client_idxs


def split_data(X, y, client_idxs, train_ratio=0.8):
    """
    Split data into train and test sets for each client
    """
    num_clients = len(client_idxs)
    train_data = []
    test_data = []

    for client_id in range(num_clients):
        client_X = X[client_idxs[client_id]]
        client_y = y[client_idxs[client_id]]

        # Split into train/test
        n_train = int(len(client_X) * train_ratio)

        train_data.append({
            'x': client_X[:n_train],
            'y': client_y[:n_train]
        })

        test_data.append({
            'x': client_X[n_train:],
            'y': client_y[n_train:]
        })

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic):
    """
    Save datasets to disk
    """
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'Size of samples for labels in clients': statistic,
    }

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + f'train{idx}_.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)

    for idx, test_dict in enumerate(test_data):
        with open(test_path + f'test{idx}_.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)

    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


def generate_cg_mnist(dir_path, num_clients=30, alpha=0.5):
    """
    Generate Colored-and-Gray MNIST dataset for multi-modal federated learning
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Download and prepare MNIST dataset
    print("Downloading and preparing MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create colored MNIST datasets
    print("Creating colored MNIST dataset...")

    # Combine training and test data for partitioning
    all_data = []
    all_labels = []

    for idx in range(len(mnist_train)):
        img, label = mnist_train[idx]
        all_data.append(img)
        all_labels.append(label)

    for idx in range(len(mnist_test)):
        img, label = mnist_test[idx]
        all_data.append(img)
        all_labels.append(label)

    all_data = torch.stack(all_data)
    all_labels = np.array(all_labels)

    # Create multi-modal dataset (grayscale + color)
    cg_mnist_data = []
    for i in range(len(all_data)):
        img = all_data[i]
        label = all_labels[i]

        # Create grayscale (1 channel)
        gray_img = img

        # Create colored version (3 channels)
        img_pil = Image.fromarray(np.uint8(img.numpy() * 255))

        # Colorize based on digit class
        if label == 0:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#FF0000")  # Red
        elif label == 1:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#00FF00")  # Green
        elif label == 2:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#0000FF")  # Blue
        elif label == 3:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#FFFF00")  # Yellow
        elif label == 4:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#FF00FF")  # Magenta
        elif label == 5:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#00FFFF")  # Cyan
        elif label == 6:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#FF8000")  # Orange
        elif label == 7:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#8000FF")  # Purple
        elif label == 8:
            colored_img = ImageOps.colorize(img_pil, "#000000", "#008080")  # Teal
        else:  # digit 9
            colored_img = ImageOps.colorize(img_pil, "#000000", "#800000")  # Maroon

        # Convert back to tensor
        colored_img = transforms.ToTensor()(colored_img)

        # Stack grayscale and colored images [4, 28, 28]
        multi_modal_img = torch.cat([gray_img.unsqueeze(0), colored_img], dim=0)
        cg_mnist_data.append(multi_modal_img)

    # Convert to numpy array
    cg_mnist_data = torch.stack(cg_mnist_data).numpy()

    print("Partitioning dataset across clients using Dirichlet distribution...")

    # Partition dataset using Dirichlet distribution
    client_idxs = dirichlet_partition(all_labels, num_clients, alpha)

    # Get statistics of label distribution
    statistic = []
    for client_id in range(num_clients):
        client_labels = all_labels[client_idxs[client_id]]
        client_stat = []
        for label in range(10):
            count = np.sum(client_labels == label)
            if count > 0:
                client_stat.append([int(label), int(count)])
        statistic.append(client_stat)

    # Split into train and test
    train_data, test_data = split_data(cg_mnist_data, all_labels, client_idxs)

    # Save the dataset
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 10, statistic)

    # Visualize some examples
    print("Visualizing examples...")
    sample_idx = random.randint(0, len(cg_mnist_data) - 1)
    sample_img = cg_mnist_data[sample_idx]
    sample_label = all_labels[sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sample_img[0], cmap='gray')
    axes[0].set_title(f"Grayscale - Label: {sample_label}")
    axes[0].axis('off')

    # Colored image (convert from CHW to HWC)
    colored = np.transpose(sample_img[1:4], (1, 2, 0))
    axes[1].imshow(colored)
    axes[1].set_title(f"Colored - Label: {sample_label}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(dir_path + "sample.png")
    plt.close()

    print(f"Generated CG-MNIST dataset with {num_clients} clients.")
    print(f"Each client has non-IID data with Dirichlet parameter alpha={alpha}")
    print(f"Dataset saved to {dir_path}")


if __name__ == "__main__":
    generate_cg_mnist("./dataset/cg_mnist/", num_clients=30, alpha=0.5)