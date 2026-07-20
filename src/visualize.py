import math
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def plot_learning_curves(history, output_dir):
    """Plot and save loss and accuracy curves from a training run.

    Args:
        history: Dictionary containing ``train_loss``, ``val_loss``,
            ``train_acc``, and ``val_acc`` sequences.
        output_dir: Directory where ``curves.png`` should be saved.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Learning Curves: Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title("Learning Curves: Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.show()


def plot_samples(config, all_images, all_labels, all_preds):
    """Plot a random sample of predictions with true and predicted labels.

    Args:
        config: Model configuration whose output directory receives the figure.
        all_images: NumPy array of normalized images in ``C x H x W`` format.
        all_labels: Array-like ground-truth class indices.
        all_preds: Array-like predicted class indices.
    """
    idxs = np.random.choice(len(all_labels), 5, replace=False)
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(idxs):
        img = np.transpose(all_images[idx], (1, 2, 0))
        img = np.clip((img * 0.5 + 0.5), 0, 1)
        true_label = CLASS_NAMES[all_labels[idx]]
        pred_label = CLASS_NAMES[all_preds[idx]]
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True : {true_label}\nPredicted : {pred_label}")
    plt.tight_layout()
    save_path = os.path.join(config.output_dir, "samples.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.show()


def plot_confusion_matrix(config, all_labels, all_preds):
    """Plot and save a CIFAR-10 confusion matrix heatmap.

    Args:
        config: Model configuration whose output directory receives the figure.
        all_labels: Array-like ground-truth class indices.
        all_preds: Array-like predicted class indices.
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(config.output_dir, "confusion.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.show()


def plot_attention_maps(config, model, test_dataset, device):
    """Plot raw images and transformer patch-importance maps.

    The visualization selects a fixed set of CIFAR-10 test images, obtains model
    features, computes patch-token norms as an importance proxy, upsamples the
    resulting grid to image resolution, and overlays it on the raw image.

    Args:
        config: Model configuration containing patch-token offset and output path.
        model: Vision transformer model with ``forward_features`` and
            ``forward_head`` methods.
        test_dataset: Transformed CIFAR-10 test dataset used for model inputs.
        device: Torch device where the model input should be evaluated.
    """
    indices = [0, 2, 74, 78, 97]
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    from torchvision import datasets as dsets
    test_dataset_raw = dsets.CIFAR10(
        root="./data", train=False, download=True, transform=None
    )

    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        img_raw, label = test_dataset_raw[idx]
        true_name = CLASS_NAMES[label]

        img_tf, _ = test_dataset[idx]
        img_tf = img_tf.unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.forward_features(img_tf)
            output = model.forward_head(features)
            pred = output.argmax(1).item()
            pred_name = CLASS_NAMES[pred]

        color = "green" if label == pred else "red"

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_raw)
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=10)
        plt.axis("off")

        patch_tokens = features[0, config.patch_token_start :]
        importance = patch_tokens.norm(dim=1).cpu().numpy()

        num_patches = importance.shape[0]
        grid = int(math.sqrt(num_patches))
        importance = importance.reshape(grid, grid)
        importance = (importance - importance.min()) / (
            importance.max() - importance.min()
        )

        attn_map = (
            F.interpolate(
                torch.tensor(importance).unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
            )
            .squeeze()
            .numpy()
        )

        plt.subplot(2, 5, i + 6)
        plt.imshow(img_raw)
        plt.imshow(attn_map, cmap="jet", alpha=0.6)
        plt.axis("off")
        plt.title("Patch Importance", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(config.output_dir, "attention.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.show()
