import os
import torch
from torch.utils.data import Dataset, DataLoader
import clip
from sklearn.model_selection import train_test_split
from config import *
from utils import  preprocess_image, preprocess_text
import pandas as pd


class RetailDataset(Dataset):
    def __init__(self, dataframe, images_path, preprocess_model, num_shots=0):
        self.df = dataframe.reset_index(drop=True)
        self.images_path = images_path
        self.preprocess_model = preprocess_model
        self.num_shots = num_shots
        
        
        if self.num_shots == 0:
            self.use_label_map = True
        else:
            self.use_label_map = False

        # Setup classes and mapping
            # 1. Get original labels
        self.original_class_names = sorted(self.df['categories'].unique().tolist())
            
            # 2. Map Original Label -> Index
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.original_class_names)}
            
            # 3. Apply LABEL_MAP if zero-shot, otherwise keep original names
        if self.use_label_map:
            self.class_names = [LABEL_MAP.get(cat, cat) for cat in self.original_class_names]
        else:
            self.class_names = self.original_class_names.copy()
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Image Preprocessing
        img_name = str(row['ImgId']) + ".jpg"
        img_path = os.path.join(self.images_path, img_name)
        image_tensor = preprocess_image(img_path, self.preprocess_model)
        
        # 2. Text Preprocessing (Depends on self.concatenate flag)
        text_token = preprocess_text(
            row.get('title', ''), 
            row.get('description', ''), 
        )
        
        # 3. Return Image, Text, and Label (if available)
        if 'categories' in self.df.columns:
            label = torch.tensor(self.cat_to_idx[row['categories']], dtype=torch.long)
            return image_tensor, text_token, label
        
        # Inference mode (Test set without labels)
        return image_tensor, text_token, row['ImgId']


def get_dataloaders(csv_path, images_path, num_shots=0, batch_size=32, model_name="ViT-L/14", train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Creates Train, Val, and Test DataLoaders. 
    Handles both Zero-shot (num_shots=0) and Few-shot (num_shots>=1) logic dynamically.
    """
    device = DEVICE
    _, preprocess_model = clip.load(model_name, device=device)
    
    df = pd.read_csv(csv_path)
    
    # 1. SPLIT DATA
    # Stratify only if 'categories' exists
    
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=random_state, 
        stratify=df['categories']
    )
    
    relative_val_size = val_size / (val_size + test_size)
    temp_stratify = temp_df['categories']
    
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=relative_val_size, 
        random_state=random_state,
        stratify=temp_stratify
    )
    
    print(f"--- SPLIT STATISTICS ---")
    print(f"Initial Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. APPLY K-SHOT LOGIC ON TRAIN SET (If num_shots >= 1)
    if num_shots > 0 and 'categories' in train_df.columns:
        train_df = train_df.groupby('categories', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), num_shots), random_state=random_state)
        ).reset_index(drop=True)

    # 3. INITIALIZE UNIFIED DATASETS
    train_dataset = RetailDataset(train_df, images_path, preprocess_model, num_shots=num_shots)
    val_dataset = RetailDataset(val_df, images_path, preprocess_model, num_shots=num_shots)
    test_dataset = RetailDataset(test_df, images_path, preprocess_model, num_shots=num_shots)

    # Sync mappings from Train to Val & Test to ensure index consistency
    val_dataset.cat_to_idx = train_dataset.cat_to_idx
    val_dataset.class_names = train_dataset.class_names
    val_dataset.original_class_names = train_dataset.original_class_names
    
    test_dataset.cat_to_idx = train_dataset.cat_to_idx
    test_dataset.class_names = train_dataset.class_names
    test_dataset.original_class_names = train_dataset.original_class_names

    # 4. INITIALIZE DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, train_dataset
