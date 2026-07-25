"""Training and validation loops."""

import time

import pandas as pd
import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train a model for one epoch and return loss and top-1 accuracy."""

    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        _, preds = torch.max(outputs, dim=1)

        running_loss += loss.item() * batch_size
        running_correct += torch.sum(preds == labels).item()
        total_samples += batch_size

        progress_bar.set_postfix(
            {"loss": running_loss / total_samples, "acc": running_correct / total_samples}
        )

    return running_loss / total_samples, running_correct / total_samples


def validate_one_epoch(model, dataloader, criterion, device):
    """Evaluate a model for one epoch and return loss and top-1 accuracy."""

    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            _, preds = torch.max(outputs, dim=1)

            running_loss += loss.item() * batch_size
            running_correct += torch.sum(preds == labels).item()
            total_samples += batch_size

            progress_bar.set_postfix(
                {"loss": running_loss / total_samples, "acc": running_correct / total_samples}
            )

    return running_loss / total_samples, running_correct / total_samples


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name):
    """Train a model and keep the best validation-accuracy checkpoint in memory."""

    history = []
    best_val_acc = 0.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} - {model_name}")
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        history.append(
            {
                "model": model_name,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        print(
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    total_time = time.time() - start_time
    print(f"\nFinished training {model_name}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {total_time / 60:.2f} minutes")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, pd.DataFrame(history)
