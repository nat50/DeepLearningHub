import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import encode_long_text_wrapper


class ZeroShotClassifier(nn.Module):
    def __init__(self, clip_model, feature_dim=512, projection_dim=256):
        super(ZeroShotClassifier, self).__init__()
        self.clip = clip_model
        
        # 1. Freeze CLIP parameters (No training for the backbone)
        for param in self.clip.parameters():
            param.requires_grad = False
    
        
    def predict(self, image, text, class_names):
        """
        Args:
            image: Preprocessed image tensor [Batch, 3, 224, 224]
            text: List of text descriptions for the batch
            class_names: List of all category names
        """
        batch_size = image.size(0)
        all_text_features = []

        # Iterate through each image in the batch
        for i in range(batch_size):
            desc = text[i] if isinstance(text[i], str) else ""
            
            # Instead of tokenizing the entire prompt cluster at once,
            # we call the wrapper function for each class individually.
            cls_feats = [encode_long_text_wrapper(self.clip, cls, desc, image.device) 
                         for cls in class_names]
            
            # Stack into shape: [Num_Classes, 512]
            feat = torch.stack(cls_feats).squeeze(1) 
            
            all_text_features.append(feat)

        # text_features shape: [Batch, Num_Classes, 512]
        text_features = torch.stack(all_text_features)
            
        # --- STEP 1: Extract features ---
        with torch.no_grad():
            image_features = self.clip.encode_image(image).float() # [Batch, 512]
            
            # --- STEP 2: L2 Normalization (Crucial before BMM) ---
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        
        # Calculate similarity scores using Batch Matrix Multiplication (BMM)
        # image_features shape: [Batch, 1, 512]
        # text_features transposed shape: [Batch, 512, Num_Classes]
        # Resulting logits shape: [Batch, Num_Classes]
        logits = torch.bmm(image_features.unsqueeze(1), text_features.transpose(1, 2)).squeeze(1)
        
        # Apply the standalone softmax function to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        return probabilities