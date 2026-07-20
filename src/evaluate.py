import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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


def predict(config, model, test_loader, device):
    """Load a saved checkpoint and predict labels for the CIFAR-10 test set.

    Args:
        config: Model configuration with checkpoint and output directory values.
        model: PyTorch model instance matching the saved checkpoint architecture.
        test_loader: Data loader yielding test ``(images, labels)`` batches.
        device: Torch device used for checkpoint loading and inference.

    Returns:
        A tuple containing normalized image tensors as a NumPy array, predicted
        class indices, and ground-truth class indices.
    """
    model_path = os.path.join(config.output_dir, config.checkpoint_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_images = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, pred = probs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu())

    all_images = torch.stack(all_images).numpy()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_images, all_preds, all_labels


def save_predictions_csv(config, all_labels, all_preds):
    """Save predicted and true CIFAR-10 labels to a CSV file.

    Args:
        config: Model configuration whose output directory receives the CSV.
        all_labels: Iterable of integer ground-truth class indices.
        all_preds: Iterable of integer predicted class indices.
    """
    df = pd.DataFrame(
        {
            "image_id": list(range(len(all_labels))),
            "true_label": [CLASS_NAMES[i] for i in all_labels],
            "predicted_label": [CLASS_NAMES[i] for i in all_preds],
        }
    )
    csv_path = os.path.join(config.output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to '{csv_path}'")


def compute_and_print_metrics(all_labels, all_preds):
    """Compute and print aggregate and per-class classification metrics.

    Args:
        all_labels: Iterable of integer ground-truth class indices.
        all_preds: Iterable of integer predicted class indices.

    Returns:
        Overall accuracy as a float in the range ``[0, 1]``.
    """
    acc = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro")
    rec_macro = recall_score(all_labels, all_preds, average="macro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print("\n=== PERFORMANCE METRICS ===")
    print(f"Overall Accuracy: {acc:.4f} ({100*acc:.2f}%)")
    print(f"Macro Precision:  {prec_macro:.4f}")
    print(f"Macro Recall:     {rec_macro:.4f}")
    print(f"Macro F1-Score:   {f1_macro:.4f}")

    prec_per = precision_score(all_labels, all_preds, average=None)
    rec_per = recall_score(all_labels, all_preds, average=None)
    f1_per = f1_score(all_labels, all_preds, average=None)

    print("\nPer-Class Metrics:")
    print("Class\t\tPrecision\tRecall\t\tF1")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<12}\t{prec_per[i]:.4f}\t\t{rec_per[i]:.4f}\t\t{f1_per[i]:.4f}"
        )

    return acc
