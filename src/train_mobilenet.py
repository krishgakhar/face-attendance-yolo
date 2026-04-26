import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 🔥 TRANSFORMS (FIXED + AUGMENTATION)
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🔥 DATASET
train_data = datasets.ImageFolder('dataset/train', transform=train_transform)
val_data = datasets.ImageFolder('dataset/val', transform=val_transform)

print("Classes:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# 🔥 MODEL
model = models.mobilenet_v2(weights="DEFAULT")

# 🔥 FREEZE BACKBONE (IMPORTANT)
for param in model.features.parameters():
    param.requires_grad = False

# 🔥 CHANGE CLASSIFIER
model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔥 LOSS + OPTIMIZER (LOW LR)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0003)

# 🔥 EARLY STOPPING
patience = 5
best_val_loss = float('inf')
counter = 0

epochs = 20

for epoch in range(epochs):

    # ===== TRAIN =====
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.2f}")

    # 🔥 EARLY STOPPING
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "models/mobilenetv2_best.pth")
        print("✅ Best model saved")

    else:
        counter += 1
        print(f"No improvement {counter}/{patience}")

        if counter >= patience:
            print("🛑 Early stopping")
            break

print("Training complete")