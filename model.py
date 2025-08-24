import os, json, glob, math, time, re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image

# Hugging Face / PyTorch
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# OCR and NER
import easyocr
import spacy
from spacy import displacy

# Mixed precision for speed
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Mount Drive (Colab)
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# -----------------------------------------
# 2) Enhanced Paths & Parameters
# -----------------------------------------
DATA_DIR = "/content/drive/MyDrive/Img"  # contains class subfolders
CACHE_DIR = "/content/drive/MyDrive/screenshot_feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 12
IMG_FEATURE_DIM = 1280  # EfficientNetV2B0 base's GAP output size
NLI_ENTAIL_IDX = 2  # [contradiction, neutral, entailment] ordering

# Enhanced feature dimensions
NSFW_DIM = 1
NLI_DIM = 0  # Will be set based on num_classes
OCR_EMBEDDING_DIM = 768  # BERT-base dimension
NER_FEATURE_DIM = 50  # Custom NER features

# -----------------------------------------
# 3) Load class names and enhanced models
# -----------------------------------------
class_names = sorted([d.name for d in Path(DATA_DIR).iterdir() if d.is_dir()])
num_classes = len(class_names)
NLI_DIM = num_classes
print("Classes:", class_names)

# Save label map
label_map_path = "/content/drive/MyDrive/screenshot_label_map.json"
with open(label_map_path, "w") as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f)
print(f"✅ Saved label map to {label_map_path}")

# -----------------------------------------
# 4) Enhanced model loading
# -----------------------------------------
print("Loading enhanced models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP for NSFW and visual understanding
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name)

# mDeBERTa MNLI for entailment
nli_name = "microsoft/mdeberta-v3-base-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)
nli_model.eval()

# Sentence transformer for OCR text embeddings
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# OCR models - multi-language support
ocr_reader = easyocr.Reader(['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'])

# NER model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# -----------------------------------------
# 5) EfficientNetV2B0 backbone (unchanged)
# -----------------------------------------
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
)
img_inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.efficientnet_v2.preprocess_input(img_inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
image_feature_extractor = tf.keras.Model(img_inputs, x)

# -----------------------------------------
# 6) Enhanced utility functions
# -----------------------------------------
def load_image_as_pil(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img

def extract_text_with_ocr(pil_image):
    """Enhanced OCR with multi-language support and confidence scoring"""
    try:
        # Convert PIL to numpy array for easyocr
        img_array = np.array(pil_image)
        results = ocr_reader.readtext(img_array)
        
        # Extract text with confidence weighting
        texts = []
        total_conf = 0
        for (bbox, text, conf) in results:
            if conf > 0.5:  # Filter low confidence detections
                texts.append(text)
                total_conf += conf
        
        combined_text = " ".join(texts)
        avg_confidence = total_conf / len(results) if results else 0
        
        return combined_text, avg_confidence, len(results)
    except Exception as e:
        print(f"OCR error: {e}")
        return "", 0.0, 0

def extract_entities_and_features(text):
    """Enhanced NER with investigative-relevant entity types"""
    if not text or len(text.strip()) == 0:
        return np.zeros(NER_FEATURE_DIM, dtype=np.float32)
    
    try:
        doc = nlp(text)
        
        # Count different entity types relevant to investigations
        entity_counts = {
            'PERSON': 0, 'ORG': 0, 'GPE': 0,  # People, orgs, locations
            'DATE': 0, 'TIME': 0, 'MONEY': 0,  # Temporal and financial
            'PRODUCT': 0, 'EVENT': 0, 'LAW': 0,  # Products, events, legal
            'EMAIL': 0, 'PHONE': 0, 'URL': 0,  # Contact info
            'PERCENT': 0, 'QUANTITY': 0, 'ORDINAL': 0,  # Numbers
            'LANGUAGE': 0, 'NORP': 0  # Language, nationalities
        }
        
        # Count entities by type
        for ent in doc.ents:
            if ent.label_ in entity_counts:
                entity_counts[ent.label_] += 1
        
        # Detect patterns using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'[\+]?[\d\-\(\)\s]{10,}'
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        entity_counts['EMAIL'] += len(re.findall(email_pattern, text))
        entity_counts['PHONE'] += len(re.findall(phone_pattern, text))
        entity_counts['URL'] += len(re.findall(url_pattern, text))
        
        # Convert to feature vector (normalized)
        features = []
        max_count = max(entity_counts.values()) if any(entity_counts.values()) else 1
        
        # Add normalized counts
        for key in sorted(entity_counts.keys()):
            features.append(entity_counts[key] / max_count)
        
        # Add text statistics
        features.extend([
            len(text) / 1000,  # Text length (normalized)
            len(text.split()) / 100,  # Word count (normalized)
            text.count('.') / 10,  # Sentence count proxy
            sum(1 for c in text if c.isupper()) / len(text) if text else 0,  # Caps ratio
            sum(1 for c in text if c.isdigit()) / len(text) if text else 0,  # Digit ratio
        ])
        
        # Pad or truncate to NER_FEATURE_DIM
        while len(features) < NER_FEATURE_DIM:
            features.append(0.0)
        features = features[:NER_FEATURE_DIM]
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        print(f"NER error: {e}")
        return np.zeros(NER_FEATURE_DIM, dtype=np.float32)

def compute_enhanced_nsfw_score(pil_image):
    """Enhanced NSFW detection with multiple prompts"""
    try:
        # Multiple NSFW detection prompts
        safety_prompts = [
            ["safe content", "inappropriate content"],
            ["work appropriate", "not work appropriate"],
            ["family friendly", "adult content"],
            ["professional", "explicit"]
        ]
        
        nsfw_scores = []
        for prompt_pair in safety_prompts:
            inputs = clip_processor(
                text=prompt_pair, 
                images=pil_image, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=1).cpu().numpy().squeeze()
            nsfw_scores.append(float(probs[1]))  # Second option is "unsafe"
        
        # Return average NSFW score
        return np.mean(nsfw_scores)
    
    except Exception as e:
        print(f"NSFW scoring error: {e}")
        return 0.5  # Neutral score on error

def compute_text_embedding(text):
    """Generate semantic embedding for OCR text"""
    if not text or len(text.strip()) == 0:
        return np.zeros(OCR_EMBEDDING_DIM, dtype=np.float32)
    try:
        embedding = sentence_model.encode(text)
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Text embedding error: {e}")
        return np.zeros(OCR_EMBEDDING_DIM, dtype=np.float32)

def compute_nli_entailment_vector(extracted_text, class_names_list):
    """Enhanced NLI with multiple hypothesis templates"""
    if extracted_text is None or len(extracted_text.strip()) == 0:
        return np.zeros(len(class_names_list), dtype=np.float32)
    
    # Multiple hypothesis templates for better coverage
    templates = [
        "This image is a {class_name} screenshot.",
        "This screenshot shows {class_name} content.",
        "The image contains {class_name} interface elements.",
        "This is a {class_name} application screen."
    ]
    
    entail_probs = []
    for c in class_names_list:
        class_scores = []
        for template in templates:
            hypothesis = template.format(class_name=c)
            
            inputs = nli_tokenizer(
                extracted_text, 
                hypothesis, 
                truncation=True, 
                padding="longest", 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                out = nli_model(**inputs).logits
                probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
                class_scores.append(float(probs[NLI_ENTAIL_IDX]))
        
        # Average across templates
        entail_probs.append(np.mean(class_scores))
    
    return np.array(entail_probs, dtype=np.float32)

def find_associated_text(image_path):
    """Find associated text file"""
    p = Path(image_path)
    txt_path = p.with_suffix(".txt")
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding="utf-8", errors="ignore")
        except:
            return ""
    return ""

# -----------------------------------------
# 7) Enhanced feature computation
# -----------------------------------------
def gather_image_paths_and_labels(root_dir):
    items = []
    for idx, cname in enumerate(class_names):
        cls_dir = Path(root_dir) / cname
        if not cls_dir.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for p in cls_dir.glob(ext):
                items.append((str(p), idx))
    return items

def precompute_enhanced_features(split_name, items_list, cache_dir=CACHE_DIR):
    """Enhanced feature computation with multi-modal fusion"""
    cache_file = Path(cache_dir) / f"{split_name}_enhanced_fused_features.npz"
    
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        data = np.load(str(cache_file))
        return data["X"], data["y"]
    
    print(f"Precomputing enhanced features for {split_name} ({len(items_list)} items)...")
    
    X_list = []
    y_list = []
    batch_images = []
    batch_metadata = []
    BATCH_PRE = 16  # Reduced batch size due to more complex processing
    
    for img_path, label in tqdm(items_list):
        try:
            pil_img = load_image_as_pil(img_path, size=IMG_SIZE)
            
            # 1. Enhanced NSFW scoring
            nsfw_score = compute_enhanced_nsfw_score(pil_img)
            
            # 2. OCR text extraction with multi-language support
            ocr_text, ocr_conf, ocr_detections = extract_text_with_ocr(pil_img)
            
            # 3. Check for associated text file
            file_text = find_associated_text(img_path)
            combined_text = f"{file_text} {ocr_text}".strip()
            
            # 4. Text embedding
            text_embedding = compute_text_embedding(combined_text)
            
            # 5. NER features
            ner_features = extract_entities_and_features(combined_text)
            
            # 6. NLI entailment scores
            nli_vec = compute_nli_entailment_vector(combined_text, class_names)
            
            # Prepare for batch image processing
            img_arr = np.array(pil_img).astype(np.float32)
            batch_images.append(img_arr)
            batch_metadata.append({
                'nsfw_score': nsfw_score,
                'text_embedding': text_embedding,
                'ner_features': ner_features,
                'nli_vec': nli_vec,
                'ocr_conf': ocr_conf,
                'ocr_detections': ocr_detections,
                'label': label
            })
            
            # Process batch
            if len(batch_images) >= BATCH_PRE:
                arr = np.stack(batch_images, axis=0)
                arr_tf = tf.convert_to_tensor(arr)
                img_feats = image_feature_extractor(arr_tf, training=False).numpy()
                
                for i, img_feat in enumerate(img_feats):
                    meta = batch_metadata[i]
                    
                    # Combine all features
                    fused = np.concatenate([
                        img_feat.astype(np.float32),  # Image features
                        np.array([meta['nsfw_score']], dtype=np.float32),  # NSFW
                        meta['text_embedding'],  # Text semantics
                        meta['ner_features'],  # NER features
                        meta['nli_vec'].astype(np.float32),  # NLI entailment
                        np.array([meta['ocr_conf'], meta['ocr_detections']], dtype=np.float32)  # OCR metadata
                    ], axis=0)
                    
                    X_list.append(fused)
                    y_list.append(meta['label'])
                
                batch_images = []
                batch_metadata = []
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Process remaining batch
    if len(batch_images) > 0:
        arr = np.stack(batch_images, axis=0)
        arr_tf = tf.convert_to_tensor(arr)
        img_feats = image_feature_extractor(arr_tf, training=False).numpy()
        
        for i, img_feat in enumerate(img_feats):
            meta = batch_metadata[i]
            fused = np.concatenate([
                img_feat.astype(np.float32),
                np.array([meta['nsfw_score']], dtype=np.float32),
                meta['text_embedding'],
                meta['ner_features'],
                meta['nli_vec'].astype(np.float32),
                np.array([meta['ocr_conf'], meta['ocr_detections']], dtype=np.float32)
            ], axis=0)
            X_list.append(fused)
            y_list.append(meta['label'])
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"Saving enhanced cache to {cache_file} (X shape: {X.shape})")
    np.savez_compressed(str(cache_file), X=X, y=y)
    
    return X, y

# -----------------------------------------
# 8) Enhanced classifier architecture
# -----------------------------------------
def build_enhanced_classifier(input_dim, num_classes, dtype_head="float32"):
    """Enhanced classifier with attention and skip connections"""
    inp = layers.Input(shape=(input_dim,), dtype="float32")
    
    # Input normalization
    x = layers.BatchNormalization()(inp)
    
    # Multi-head attention for feature importance
    x_reshaped = layers.Reshape((1, input_dim))(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=8, key_dim=input_dim // 8
    )(x_reshaped, x_reshaped)
    x_attended = layers.Flatten()(attention_output)
    
    # Residual connection
    x_combined = layers.Add()([x, x_attended])
    
    # Deep feature extraction with skip connections
    x1 = layers.Dense(1024, activation="relu")(x_combined)
    x1 = layers.Dropout(0.3)(x1)
    x1 = layers.BatchNormalization()(x1)
    
    x2 = layers.Dense(512, activation="relu")(x1)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.BatchNormalization()(x2)
    
    # Skip connection
    x1_proj = layers.Dense(512, activation="relu")(x1)
    x2_skip = layers.Add()([x2, x1_proj])
    
    x3 = layers.Dense(256, activation="relu")(x2_skip)
    x3 = layers.Dropout(0.2)(x3)
    x3 = layers.BatchNormalization()(x3)
    
    x4 = layers.Dense(128, activation="relu")(x3)
    x4 = layers.Dropout(0.2)(x4)
    
    # Output layer
    out = layers.Dense(num_classes, activation="softmax", dtype=dtype_head)(x4)
    
    model = keras.Model(inp, out)
    return model

# -----------------------------------------
# 9) Main execution
# -----------------------------------------
if __name__ == "__main__":
    # Gather data
    all_items = gather_image_paths_and_labels(DATA_DIR)
    all_items_sorted = sorted(all_items)
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(all_items_sorted))
    all_perm = [all_items_sorted[i] for i in perm]
    split_at = int(0.8 * len(all_perm))
    train_items = all_perm[:split_at]
    val_items = all_perm[split_at:]
    
    # Compute enhanced features
    X_train, y_train = precompute_enhanced_features("train", train_items)
    X_val, y_val = precompute_enhanced_features("val", val_items)
    
    print("Enhanced feature shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    
    # Build and train enhanced model
    feature_dim = X_train.shape[1]
    dtype_head = "float32" if tf.keras.mixed_precision.global_policy().name == "mixed_float16" else None
    enhanced_model = build_enhanced_classifier(feature_dim, num_classes, dtype_head=dtype_head)
    
    enhanced_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", "top_3_accuracy"]
    )
    
    enhanced_model.summary()
    
    # Training with enhanced callbacks
    checkpoint_path = "/content/drive/MyDrive/enhanced_screenshot_classifier.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor="val_accuracy",
            save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(
            patience=7, 
            restore_best_weights=True,
            monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    history = enhanced_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )
    
    # Enhanced evaluation
    y_pred_probs = enhanced_model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, 
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Enhanced Multi-Modal Classifier - Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Save model and metadata
    enhanced_model.save(checkpoint_path)
    print(f"✅ Saved enhanced model to {checkpoint_path}")
    
    # Save feature information
    feature_info = {
        'total_feature_dim': feature_dim,
        'img_feature_dim': IMG_FEATURE_DIM,
        'nsfw_dim': NSFW_DIM,
        'text_embedding_dim': OCR_EMBEDDING_DIM,
        'ner_feature_dim': NER_FEATURE_DIM,
        'nli_dim': NLI_DIM,
        'ocr_metadata_dim': 2,
        'class_names': class_names
    }
    
    with open("/content/drive/MyDrive/enhanced_feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Feature enhancement complete!")
