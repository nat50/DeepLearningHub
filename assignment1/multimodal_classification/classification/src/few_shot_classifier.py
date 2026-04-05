import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *
import clip
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from dataset import get_dataloaders

class FewshotClassifier(nn.Module):
    def __init__(self, model, num_classes):
        super(FewshotClassifier, self).__init__()
        self.model = model
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Projection layers to reduce dimensionality
        
        
        projection_dim = self.model.visual.output_dim  
        self.layer_norm = nn.LayerNorm(projection_dim * 2)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, text):
        # 1. Feature Extraction (Frozen)
        with torch.no_grad():
            image_features = self.model.encode_image(image).float()
            text_features = self.model.encode_text(text).float()
        
        # 2. L2 Normalization (Crucial for CLIP features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        
        # 3. Concatenate: [Batch, 768] + [Batch, 768] -> [Batch, 1536]
        combined = torch.cat((image_features, text_features), dim=1)
        
        # 4. Normalize concatenated features
        combined = self.layer_norm(combined)
        
        # 5. Final Classification
        logits = self.classifier(combined)
        return logits


    def fit(self, train_loader, val_loader):
        """
        Trains the Few-shot model with validation and early stopping.
        Hyperparameters and settings are loaded directly from config.py.
        Saves the learning curves (loss and accuracy) to an image.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        best_val_acc = 0.0
        patience_counter = 0

        # Dictionary to store training history for plotting
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

        for epoch in range(EPOCHS):
            # --- TRAIN PHASE ---
            self.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
            for images, texts, labels in train_loop:
                images, texts, labels = images.to(DEVICE), texts.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self(images, texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Accumulate total loss
                train_loss += loss.item()
                
                # Calculate number of correct predictions on the training set
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                train_loop.set_postfix(loss=loss.item())
            
            # Average Loss and Accuracy for the Training Set
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = 100 * correct_train / total_train

            # --- VALIDATION PHASE ---
            val_loss, val_acc, _ = self.predict(val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save to history
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(avg_train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # --- EARLY STOPPING & SAVING ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                torch.save(self.state_dict(), SAVE_MODEL_PATH)
                print(f"  --> Saved Best Model (Val Acc: {val_acc:.2f}%) to {SAVE_MODEL_PATH}")
            else:
                patience_counter += 1
                print(f"  --> EarlyStopping counter: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"\n[Early Stopping] Training halted. Validation accuracy did not improve for {PATIENCE} epochs.")
                break
        
        # ==========================================
        # PLOT LEARNING CURVES
        # ==========================================
        print("\n--- Generating Learning Curves ---")
        actual_epochs = range(1, len(history["train_loss"]) + 1)
        
        plt.figure(figsize=(14, 5))

        # 1. Plot Loss Graph
        plt.subplot(1, 2, 1)
        plt.plot(actual_epochs, history["train_loss"], label='Train Loss', marker='o')
        plt.plot(actual_epochs, history["val_loss"], label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 2. Plot Accuracy Graph
        plt.subplot(1, 2, 2)
        plt.plot(actual_epochs, history["train_acc"], label='Train Accuracy', marker='o')
        plt.plot(actual_epochs, history["val_acc"], label='Validation Accuracy', marker='o')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        # Save image using the path from config
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH)
        plt.close() # Close plot to free up memory
        print(f"--> Saved training charts to: {PLOT_SAVE_PATH}")

    def predict(self, image_path, text_description, clip_preprocess, class_names=None):
        """
        Inference Pipeline cho 1 mẫu dữ liệu duy nhất.
        Nhận vào dữ liệu thô (Raw Image, Raw Text) -> Xử lý -> Trả về kết quả dự đoán.
        """
        self.eval() # Chuyển model sang chế độ đánh giá
        
        # Lấy device hiện tại của model
        device = next(self.parameters()).device 
        
        # ==========================================
        # BƯỚC 1: XỬ LÝ ẢNH (Image Preprocessing)
        # ==========================================
        try:
            image = Image.open(image_path).convert("RGB")
            # Xử lý ảnh và thêm chiều Batch (shape: [1, 3, 224, 224])
            image_tensor = clip_preprocess(image).unsqueeze(0).to(device) 
        except Exception as e:
            raise ValueError(f"Không thể đọc hoặc xử lý ảnh tại {image_path}. Lỗi: {e}")

        # ==========================================
        # BƯỚC 2: XỬ LÝ TEXT (Text Preprocessing)
        # ==========================================
        # Đảm bảo text không bị quá 77 tokens của CLIP
        # (Nếu bạn có hàm clean_text riêng, hãy gọi nó ở đây trước khi tokenize)
        clean_text = str(text_description).lower() if text_description else "product"
        text_tensor = clip.tokenize(clean_text, truncate=True).to(device)

        # ==========================================
        # BƯỚC 3: DỰ ĐOÁN (Model Inference)
        # ==========================================
        with torch.no_grad():
            logits = self(image_tensor, text_tensor)
            
            # Áp dụng softmax để chuyển logit thành phần trăm xác suất (0-1)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
            # Lấy index và xác suất cao nhất
            top_prob, top_idx = torch.max(probs, dim=0)

        # ==========================================
        # BƯỚC 4: HẬU XỬ LÝ KẾT QUẢ (Post-processing)
        # ==========================================
        predicted_idx = top_idx.item()
        confidence = top_prob.item() * 100 # Đổi ra %
        
        result = {
            "predicted_idx": predicted_idx,
            "confidence": f"{confidence:.2f}%",
        }
        
        # Nếu truyền vào danh sách tên class, tự động map index sang tên class
        if class_names:
            result["predicted_class"] = class_names[predicted_idx]
            
        return result
    
if __name__ == "__main__":
    """
    Main training script for FewshotClassifier model.
    Loads CLIP model, creates dataloaders, and trains the classifier.
    """
    print("=" * 70)
    print("FEWSHOT CLASSIFIER - TRAINING SCRIPT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch Size: {TRAIN_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 70)

    # =========================================================
    # 1. LOAD CLIP MODEL
    # =========================================================
    print("\n--- 1. Loading CLIP model ---")
    print(f"Loading: {MODEL_NAME}")
    clip_model, _ = clip.load(MODEL_NAME, device=DEVICE)
    print("✓ CLIP model loaded successfully")

    # =========================================================
    # 2. GET FEATURE DIMENSION FROM CLIP
    # =========================================================
    print("\n--- 2. Extracting model dimensions ---")
    feature_dim = clip_model.visual.output_dim
    print(f"Feature dimension: {feature_dim}")

    # =========================================================
    # 3. CREATE DATALOADERS
    # =========================================================
    print("\n--- 3. Preparing dataloaders ---")
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        csv_path=DATASET_CSV,
        images_path=IMAGES_DIR,
        num_shots=NUM_SHOTS,
        batch_size=TRAIN_BATCH_SIZE,
        model_name=MODEL_NAME,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    num_classes = len(train_dataset.class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # =========================================================
    # 4. INITIALIZE CLASSIFIER
    # =========================================================
    print("\n--- 4. Initializing FewshotClassifier ---")
    classifier = FewshotClassifier(
        model=clip_model,
        num_classes=num_classes
    ).to(DEVICE)
    print("✓ Classifier created successfully")

    # Print model info
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.3f}M")

    # =========================================================
    # 5. TRAIN THE MODEL
    # =========================================================
    print("\n--- 5. Starting training ---")
    print(f"Training for {EPOCHS} epochs with early stopping (patience={PATIENCE})")
    classifier.fit(train_loader, val_loader)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Model saved to: {SAVE_MODEL_PATH}")
    print(f"Training curves saved to: {PLOT_SAVE_PATH}")