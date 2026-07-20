import time
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score


def evaluate_loader(model, loader, criterion, device):
    """Evaluate a model over a data loader.

    Args:
        model: PyTorch model to evaluate.
        loader: Data loader yielding ``(images, labels)`` batches.
        criterion: Loss function used to compute average loss.
        device: Torch device where inputs and labels should be evaluated.

    Returns:
        A tuple containing average loss, accuracy, predicted class indices, and
        ground-truth class indices.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, all_preds, all_labels


def train_model(config, model, train_loader, test_loader, device, num_epochs=20):
    """Fine-tune a model classification head and save the best checkpoint.

    The function trains with cross-entropy loss using label smoothing, AdamW,
    and a cosine annealing learning-rate schedule. Validation accuracy is
    evaluated at the end of each epoch; whenever it improves, the model state is
    saved to ``config.output_dir/config.checkpoint_name``.

    Args:
        config: Model configuration with output directory and checkpoint name.
        model: PyTorch model to train.
        train_loader: Data loader for training batches.
        test_loader: Data loader used for validation after each epoch.
        device: Torch device where tensors should be moved.
        num_epochs: Number of full training passes over ``train_loader``.

    Returns:
        A tuple containing the training history dictionary and the best
        validation accuracy reached during training.
    """
    best_val_acc = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print("Starting training...\n")
    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        val_loss, val_acc, _, _ = evaluate_loader(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(config.output_dir, config.checkpoint_name)
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1}: New best accuracy! Saved model.")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f} | "
            f"LR: {current_lr:.2e}\n"
        )

    total_time = time.time() - total_start_time
    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}")
    print(f"Total training time: {total_time/60:.2f} minutes")

    return history, best_val_acc
