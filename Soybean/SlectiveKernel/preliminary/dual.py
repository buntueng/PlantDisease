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
# UPDATED: Changed output directory to reflect EfficientNetV2-S + Dual Attention
OUTPUT_DIR = '/home/bt/Desktop/Bee/soybean/modelv3/output/efficientnet_v2_s_dual_attention_10fold'
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = 8
NUM_FOLDS = 10
LEARNING_RATE = 0.0001 

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dual Attention Components ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Using Conv2d as an MLP (shared network)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class DualAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(DualAttention, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Sequential application of channel and spatial attention
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- Custom Model Wrapper ---
class EfficientNetV2_DualAttention(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetV2_DualAttention, self).__init__()
        # Load base model with default best weights
        base_model = models.efficientnet_v2_s(weights='DEFAULT')
        
        self.features = base_model.features
        
        # EfficientNetV2-S outputs 1280 channels from its final feature block
        self.dual_attention = DualAttention(in_planes=1280) 
        
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        
        # Modify the final classification layer
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dual_attention(x)  # Apply Dual Attention 
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
print(f"Starting {NUM_FOLDS}-Fold Cross Validation [EfficientNetV2-S + Dual Attention] on {device}...")

# Iterate through Folds
for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f"\n--- Fold {fold + 1}/{NUM_FOLDS} ---")
    
    train_sub = Subset(full_dataset, train_ids)
    val_sub = Subset(full_dataset, val_ids)
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)

    # UPDATED: Initialize Custom EfficientNetV2-S + Dual Attention Model
    model = EfficientNetV2_DualAttention(num_classes=NUM_CLASSES)
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
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'efficientnet_v2_s_dual_att_fold{fold+1}_best.pth'))
            if epoch_val_acc > overall_best_acc:
                overall_best_acc = epoch_val_acc
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'efficientnet_v2_s_dual_att_model.pth'))

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
results_df.to_csv(os.path.join(OUTPUT_DIR, 'efficientnet_v2_s_dual_att_results.csv'), index=False)

history_df = pd.DataFrame(history_list, columns=['Fold', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time (s)'])
# UPDATED: File save names
history_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history_efficientnet_v2_s_dual_att.csv'), index=False)

print(f"Process Complete. All files saved to: {OUTPUT_DIR}")