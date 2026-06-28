import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, confusion_matrix, 
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import label_binarize

# --- Configuration ---
DATASET_PATH = '/home/bt/Desktop/Bee/soybean/balanced_dataset_224'
# UPDATED: Changed output directory to reflect EfficientNetV2-S + Shuffle Attention
OUTPUT_DIR = '/home/bt/Desktop/Bee/soybean/modelv3/output/efficientnet_v2_s_shuffle_attention_10fold'
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = 8
NUM_FOLDS = 10
LEARNING_RATE = 0.0001 

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Shuffle Attention Components ---
class ShuffleAttention(nn.Module):
    def __init__(self, channel=1280, G=8):
        super(ShuffleAttention, self).__init__()
        self.G = G
        self.channel = channel
        self.sub_channel = channel // (2 * G)
        
        # Channel Attention parameters
        self.weight_c = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        self.bias_c = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        
        # Spatial Attention parameters
        self.weight_s = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        self.bias_s = nn.Parameter(torch.zeros(1, self.sub_channel, 1, 1))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(self.sub_channel, self.sub_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Group into subfeatures
        x = x.view(b * self.G, -1, h, w) 
        
        # Split into two branches
        x_0, x_1 = x.chunk(2, dim=1) 
        
        # Branch 1: Channel Attention
        x_c = self.avg_pool(x_0)
        x_c = self.weight_c * x_c + self.bias_c
        x_0 = x_0 * self.sigmoid(x_c)
        
        # Branch 2: Spatial Attention
        x_s = self.gn(x_1)
        x_s = self.weight_s * x_s + self.bias_s
        x_1 = x_1 * self.sigmoid(x_s)
        
        # Concatenate
        out = torch.cat([x_0, x_1], dim=1) 
        out = out.view(b, -1, h, w)
        
        # Channel Shuffle
        out = self.channel_shuffle(out, 2)
        return out

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.shape
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

# --- Custom Model Wrapper ---
class EfficientNetV2_ShuffleAttention(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetV2_ShuffleAttention, self).__init__()
        # Load base model with default best weights
        base_model = models.efficientnet_v2_s(weights='DEFAULT')
        
        self.features = base_model.features
        
        # EfficientNetV2-S outputs 1280 channels from its final feature block
        self.shuffle_attention = ShuffleAttention(channel=1280, G=8) 
        
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        
        # Modify the final classification layer
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.shuffle_attention(x)  # Apply Shuffle Attention
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- Data Transformations ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load Dataset ---
full_dataset = datasets.ImageFolder(DATASET_PATH, transform=data_transforms)
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

results_list = []
history_list = []
overall_best_acc = 0.0

# UPDATED: Print statement
print(f"Starting {NUM_FOLDS}-Fold Cross Validation [EfficientNetV2-S + Shuffle Attention] on {device}...")

# Iterate through Folds
for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f"\n--- Fold {fold + 1}/{NUM_FOLDS} ---")
    
    train_sub = Subset(full_dataset, train_ids)
    val_sub = Subset(full_dataset, val_ids)
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)

    # UPDATED: Initialize Custom EfficientNetV2-S + Shuffle Attention Model
    model = EfficientNetV2_ShuffleAttention(num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_fold_acc = 0.0
    fold_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # --- Training Phase ---
        model.train()
        train_loss, train_correct = 0.0, 0
        train_bar = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1} [Train]")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_bar.set_postfix(loss=loss.item())

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_labels, all_preds, all_probs = [], [], []
        val_bar = tqdm(val_loader, desc=f"Fold {fold+1} Ep {epoch+1} [Val]")
        
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                val_correct += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Logging Metrics
        epoch_train_acc = train_correct.double() / len(train_ids)
        epoch_val_acc = val_correct.double() / len(val_ids)
        epoch_time = time.time() - epoch_start
        
        print(f"Summary Fold {fold+1} Epoch {epoch+1}: Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} | Time: {epoch_time:.2f}s")

        history_list.append([fold+1, epoch+1, train_loss/len(train_ids), epoch_train_acc.item(), 
                             val_loss/len(val_ids), epoch_val_acc.item(), epoch_time])

        # Save Best Model Logic
        if epoch_val_acc > best_fold_acc:
            best_fold_acc = epoch_val_acc
            # UPDATED: File save names
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'efficientnet_v2_s_shuffle_att_fold{fold+1}_best.pth'))
            if epoch_val_acc > overall_best_acc:
                overall_best_acc = epoch_val_acc
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'efficientnet_v2_s_shuffle_att_model.pth'))

    # --- Calculate Final Fold Metrics ---
    fold_time = time.time() - fold_start_time
    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).sum()
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tn = cm.sum() - (fp + fn + np.diag(cm))
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    sen = recall_score(y_true, y_pred, average='macro')
    pre = precision_score(y_true, y_pred, average='macro')
    spec = (tn.sum() / (tn.sum() + fp.sum()))
    mcc = matthews_corrcoef(y_true, y_pred)
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr')

    results_list.append([fold+1, acc, f1, sen, pre, spec, mcc, auc, tp, tn.sum(), fp.sum(), fn.sum(), fold_time])

# --- Final Save ---
results_df = pd.DataFrame(results_list, columns=['Fold', 'Accuracy', 'F1 Score', 'Sensitivity', 'Precision', 'Specificity', 'MCC', 'AUC', 'TP', 'TN', 'FP', 'FN', 'Training Time (s)'])
# UPDATED: File save names
results_df.to_csv(os.path.join(OUTPUT_DIR, 'efficientnet_v2_s_shuffle_att_results.csv'), index=False)

history_df = pd.DataFrame(history_list, columns=['Fold', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time (s)'])
# UPDATED: File save names
history_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history_efficientnet_v2_s_shuffle_att.csv'), index=False)

print(f"Process Complete. All files saved to: {OUTPUT_DIR}")