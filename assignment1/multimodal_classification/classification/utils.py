import string
import re
from turtle import title
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
import torch
import clip

stop_words = set(stopwords.words('english'))



def clean_text(text):
    """
    Replicates the get_token and clean_text functions from your notebook:
    - Lowercase, remove punctuation, remove stopwords, and remove words shorter than 2 characters.
    """
    if not isinstance(text, str) or text == "":
        return "product"
    
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return " ".join(cleaned_words)

def preprocess_text(title, description):
    
    text = f"{str(title)} {str(description)}"
        
    text = clean_text(text)
    
    return clip.tokenize(text if text else "product", truncate=True)[0]

def preprocess_image(image_path, clip_preprocess_fn):
    try:
        image = Image.open(image_path).convert("RGB")
        return clip_preprocess_fn(image)
    except Exception as e:
       
        return torch.zeros(3, 224, 224)
    
def encode_long_text_wrapper(model, category, description, device):
        # Chia nhỏ description thành các cụm ~60 từ để đảm bảo + category vẫn < 77 tokens
        words = description.split()
        chunk_size = 60
        str_chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        if not str_chunks: str_chunks = [""] # Case description rỗng

        chunk_features = []
        for s in str_chunks:
            prompt = f"Category: {category}. {s}"
            # Tokenize từng đoạn nhỏ
            tokens = clip.tokenize(prompt, truncate=True).to(device)
            with torch.no_grad():
                feat = model.encode_text(tokens).float()
                feat /= feat.norm(dim=-1, keepdim=True)
                chunk_features.append(feat)

        # Lấy trung bình cộng (logic y hệt hàm của bạn)
        final_feat = torch.mean(torch.stack(chunk_features), dim=0)
        return final_feat / final_feat.norm(dim=-1, keepdim=True)