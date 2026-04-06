import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import encode_long_text_wrapper


class ZeroShotClassifier(nn.Module):
    def __init__(self, clip_model):
        super(ZeroShotClassifier, self).__init__()
        self.clip = clip_model
        
        # 1. Freeze CLIP parameters (No training for the backbone)
        for param in self.clip.parameters():
            param.requires_grad = False
    
        
    def forward(self, image, text, class_names):
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
            
            # Stack into shape: [Num_Classes, 768]
            feat = torch.stack(cls_feats).squeeze(1) 
            
            all_text_features.append(feat)

        # text_features shape: [Batch, Num_Classes, 768]
        text_features = torch.stack(all_text_features)
            
        # --- STEP 1: Extract features ---
        with torch.no_grad():
            image_features = self.clip.encode_image(image).float() # [Batch, 768]
            
            # --- STEP 2: L2 Normalization (Crucial before BMM) ---
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        
        # Calculate similarity scores using Batch Matrix Multiplication (BMM)
        # image_features shape: [Batch, 1, 768]
        # text_features transposed shape: [Batch, 768, Num_Classes]
        # Resulting logits shape: [Batch, Num_Classes]
        value = torch.bmm(image_features.unsqueeze(1), text_features.transpose(1, 2)).squeeze(1)
        
       
        
        return value
    
    def predict(self, value, class_names):
        """
        Wrapper function to return predicted class labels instead of probabilities.
        """
        probabilities = F.softmax(value, dim=-1)
        preds = torch.argmax(probabilities, dim=-1)
        preds = [class_names[idx] for idx in preds.cpu().numpy()]
        return probabilities, preds