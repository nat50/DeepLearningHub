import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from dataset import get_dataloaders
from config import *
import clip


def count_parameters(model):
    """Calculate total and trainable parameters of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_model_size_mb(model):
    """Estimate model memory usage in RAM."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 ** 2)


def get_checkpoint_size_mb(model_path):
    """Get checkpoint file size on disk."""
    if model_path is not None and os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 ** 2)
    return None


def evaluate_model(model, dataloader, device, class_names, model_path=None, save_cm_path="test_confusion_matrix.png"):
    """
    Evaluate model on dataloader with detailed metrics.
    
    Args:
        model: Trained model instance
        dataloader: DataLoader to evaluate
        device: 'cuda' or 'cpu'
        class_names: List of category names
        model_path: Path to checkpoint model (for displaying size)
        save_cm_path: Path to save confusion matrix image
    
    Returns:
        Dictionary containing all evaluation results
    """
    model.eval()
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    trainable_ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0
    ram_model_size_mb = estimate_model_size_mb(model)
    disk_model_size_mb = get_checkpoint_size_mb(model_path)

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Total parameters     : {total_params / 1e6:.2f} M")
    print(f"Trainable parameters : {trainable_params / 1e6:.3f} M")
    print(f"Trainable ratio      : {trainable_ratio:.2f}%")
    print(f"Estimated RAM size   : {ram_model_size_mb:.2f} MB")
    if disk_model_size_mb is not None:
        print(f"Checkpoint size      : {disk_model_size_mb:.2f} MB")

    all_preds = []
    all_labels = []
    total_samples = 0
    total_batches = 0

    print(f"\n--- Starting evaluation on {len(dataloader.dataset)} samples ---")


    start_time = time.time()

    with torch.no_grad():
        for images, texts, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            batch_size_now = images.size(0)

            if device == "cuda":
                torch.cuda.synchronize()

            # Forward pass first to get logits
            logits = model(images, texts, class_names)
            
            # Then use predict to get probabilities and predicted labels
            probabilities, predicted_labels = model.predict(logits, class_names)
            # Convert predicted labels (class names) back to indices
            preds = torch.tensor([class_names.index(label) for label in predicted_labels], device=device)

            if device == "cuda":
                torch.cuda.synchronize()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_samples += batch_size_now
            total_batches += 1

    end_time = time.time()

    total_inference_time = end_time - start_time
    avg_time_per_sample = total_inference_time / total_samples if total_samples > 0 else 0.0
    avg_time_per_batch = total_inference_time / total_batches if total_batches > 0 else 0.0
    throughput = total_samples / total_inference_time if total_inference_time > 0 else 0.0

    # Metrics
    acc = accuracy_score(all_labels, all_preds)

    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n--- Overall Classification Metrics ---")
    print(f"Accuracy           : {acc * 100:.2f}%")
    print(f"Macro Precision    : {precision_macro * 100:.2f}%")
    print(f"Macro Recall       : {recall_macro * 100:.2f}%")
    print(f"Macro F1-score     : {f1_macro * 100:.2f}%")
    print(f"Weighted Precision : {precision_weighted * 100:.2f}%")
    print(f"Weighted Recall    : {recall_weighted * 100:.2f}%")
    print(f"Weighted F1-score  : {f1_weighted * 100:.2f}%")

    print("\n--- Inference Performance ---")
    print(f"Total inference time  : {total_inference_time:.4f} seconds")
    print(f"Average time / sample : {avg_time_per_sample * 1000:.4f} ms")
    print(f"Average time / batch  : {avg_time_per_batch:.4f} seconds")
    print(f"Throughput            : {throughput:.2f} samples/second")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=2,
            zero_division=0
        )
    )

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_cm_path, dpi=300, bbox_inches='tight')
    print(f"\n--> Saved Confusion Matrix image to '{save_cm_path}'")
    plt.close()

    return {
        "accuracy": acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "total_inference_time": total_inference_time,
        "avg_time_per_sample_ms": avg_time_per_sample * 1000,
        "avg_time_per_batch_s": avg_time_per_batch,
        "throughput": throughput,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "ram_model_size_mb": ram_model_size_mb,
        "disk_model_size_mb": disk_model_size_mb,
    }


if __name__ == "__main__":
    """
    Script to evaluate Zero-shot or Few-shot model on test set.
    Supports both model types via configuration below.
    """
    # =========================================================
    # 1. MAIN CONFIGURATION
    # =========================================================
    BATCH_SIZE = TRAIN_BATCH_SIZE
    # Example: MODEL_PATH = "model/zeroshot_clip.pth"
    # Example: MODEL_PATH = "model/fewshot_clip.pth"

    print("=" * 70)
    print("EVALUATION SCRIPT - MULTIMODAL CLASSIFICATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Shots: {NUM_SHOTS} {'(Zero-shot)' if NUM_SHOTS == 0 else f'({NUM_SHOTS}-shot)'}")
    print("=" * 70)

    # =========================================================
    # 2. DATALOADER CONFIGURATION
    # =========================================================
    print("\n--- 1. Preparing DataLoader ---")
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        csv_path=DATASET_CSV,
        images_path=IMAGES_DIR,
        num_shots=NUM_SHOTS,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=42
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # =========================================================
    # 3. LOAD CLIP MODEL
    # =========================================================
    print(f"\n--- 2. Loading CLIP model: {MODEL_NAME} ---")
    clip_model, _ = clip.load(MODEL_NAME, device=DEVICE)

    # =========================================================
    # 4. INITIALIZE/LOAD MODEL
    # =========================================================
    print(f"\n--- 3. Initializing model ({'Zero-shot' if NUM_SHOTS == 0 else f'{NUM_SHOTS}-shot Few-shot'}) ---")
    
    if NUM_SHOTS == 0:
        # ZERO-SHOT MODEL
        print("Using: ZeroShotClassifier")
        from src.zero_shot_classifier import ZeroShotClassifier
        model = ZeroShotClassifier(
            clip_model=clip_model
        ).to(DEVICE)
    else:
        # FEW-SHOT MODEL
        print(f"Using: FewshotClassifier ({NUM_SHOTS}-shot)")
        from src.few_shot_classifier import FewshotClassifier
        model = FewshotClassifier(
            model=clip_model,
            num_classes=num_classes
        ).to(DEVICE)

    # Load checkpoint if available
    if SAVE_MODEL_PATH is not None and os.path.exists(SAVE_MODEL_PATH):
        print(f"Loading checkpoint from: {SAVE_MODEL_PATH}")
        checkpoint = torch.load(SAVE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded successfully")
    
    # =========================================================
    # 5. EVALUATE MODEL
    # =========================================================
    print(f"\n--- 4. Starting evaluation on Test set ---")
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        class_names=class_names,
        model_path=SAVE_MODEL_PATH,
        save_cm_path="evaluate/config/test_confusion_matrix.png"
    )

    # =========================================================
    # 6. SUMMARY OF RESULTS
    # =========================================================
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {results['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1: {results['weighted_f1'] * 100:.2f}%")
    print(f"Throughput: {results['throughput']:.2f} samples/second")
    print(f"Average time per sample: {results['avg_time_per_sample_ms']:.4f} ms")
    print("=" * 70)