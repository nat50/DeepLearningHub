import torch
import os
from pathlib import Path

# Get the current file directory
CURRENT_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Use absolute paths based on current directory
NUM_SHOTS = 1
DATASET_CSV = str(CURRENT_DIR / "data" /"train" /"train.csv")
IMAGES_DIR = str(CURRENT_DIR / "data" / "train"/"train")
SAVE_MODEL_PATH = str(CURRENT_DIR / "model" / f"fewshot_clip_shots{NUM_SHOTS}.pth")
PLOT_SAVE_PATH = str(CURRENT_DIR / "evaluate" / "config" / "training_charts.png")
# ==========================================
# 2. HARDWARE & MODEL CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-L/14"
FEATURE_DIM = 512       # Default output vector size for ViT-B/32
PROJECTION_DIM = 256    # Dimensionality reduction size for Few-shot/Zero-shot

# ==========================================
# 3. TRAINING HYPERPARAMETERS
# ==========================================
EPOCHS = 10
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16    # Separate batch sizes for train and eval if needed
LEARNING_RATE = 1e-3
PATIENCE = 5            # Number of epochs for early stopping

# ==========================================
# 4. DATALOADER & FEW-SHOT CONFIGURATION
# ==========================================
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==========================================
# 5. LABEL DESCRIPTION DICTIONARY (LABEL MAP)
# ==========================================
LABEL_MAP = {
    "All Beauty": "Professional salon hair care, luxury perfumes, essential oils, and premium cosmetics",
    "Beauty": "Daily personal care, hair styling tools, tanning lotions, and grooming products",
    "Health & Personal Care": "Vitamins, dietary supplements, over-the-counter medicine, and wellness products",
    "All Electronics": "Computer cables, blank media like DVDs, replacement batteries, and tech accessories",
    "Electronics": "Consumer electronics, portable radios, scanners, and office electronic supplies",
    "Cell Phones & Accessories": "Mobile phone screen protectors, cell phone cases, chargers, and handset accessories",
    "Baby": "Baby furniture, cribs, strollers, and large infant travel gear",
    "Baby Products": "Baby diapers, nursing pads, toddler toys, and everyday infant care accessories",
    "Appliances": "Kitchen appliances, ice makers, cooktop accessories, and large household machines",
    "Tools & Home Improvement": "Power tools, bathroom lighting fixtures, saw blades, and home repair hardware",
    "Patio, Lawn & Garden": "Outdoor cooking tools, hydroponic supplies, gardening gear, and patio accessories",
    "Pet Supplies": "Pet feeding bowls, aquarium breeding nets, animal grooming powder, and pet accessories",
    "Arts, Crafts & Sewing": "Crafting tools, sewing supplies, painting kits, and DIY art materials",
    "Automotive": "Car floor mats, automotive seat covers, vehicle accessories, and car care products",
    "Clothing, Shoes & Jewelry": "Apparel, wristwatches, Halloween costumes, footwear, and fashion accessories",
    "Grocery & Gourmet Food": "Packaged food, tea bags, cooking ingredients, and grocery snacks",
    "Industrial & Scientific": "Industrial hardware, strong adhesives, lab equipment, and specialized scientific tools",
    "Musical Instruments": "Drum accessories, speaker enclosures, guitars, and music performance gear",
    "Office Products": "Office supplies, printer ink cartridges, wall calendars, and classroom bulletin boards",
    "Sports & Outdoors": "Sports team apparel, weightlifting equipment, outdoor hunting gear, and fitness items",
    "Toys & Games": "Remote control cars, puzzles, children's toys, and board games"
}