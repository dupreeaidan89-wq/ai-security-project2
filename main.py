import torch import torch.nn as nn import torch.optim as optim import torchvision import torchvision.transforms as transforms import matplotlib.pyplot as plt device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Load MNIST transform = transforms.Compose([transforms.ToTensor()]) train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True) # Neural Network class Net(nn.Module): def __init__(self): super(Net, self).__init__() self.fc1 = nn.Linear(28*28, 128) self.fc2 = nn.Linear(128, 10) def forward(self, x): x = x.view(-1, 28*28) x = torch.relu(self.fc1(x)) x = self.fc2(x) return x model = Net().to(device) criterion = nn.CrossEntropyLoss() optimizer = optim.Adam(model.parameters(), lr=0.001) # Train Model def train(): model.train() for epoch in range(3): for data, target in train_loader: data, target = data.to(device), target.to(device) optimizer.zero_grad() output = model(data) loss = criterion(output, target) loss.backward() optimizer.step() print(f"Epoch {epoch+1} complete") # Test Accuracy def test(): model.eval() correct = 0 total = 0 for data, target in test_loader: data, target = data.to(device), target.to(device) output = import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------
# 1. Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# -------------------------
# 2. CNN Model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# -------------------------
# 3. Train Model
# -------------------------
def train(model, loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -------------------------
# 4. Evaluate Model
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

# -------------------------
# 5. FGSM Attack
# -------------------------
def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * gradient.sign()
    adv_image = image + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image

def evaluate_fgsm(model, loader, epsilon=0.25):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        gradient = images.grad.data
        adv_images = fgsm_attack(images, epsilon, gradient)

        adv_outputs = model(adv_images)
        preds = adv_outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"FGSM Accuracy (epsilon={epsilon}): {acc*100:.2f}%")
    return acc

# -------------------------
# 6. Defense (Adversarial Training)
# -------------------------
def adversarial_train(model, loader, epochs=2, epsilon=0.25):
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True

            outputs = model(images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            gradient = images.grad.data
            adv_images = fgsm_attack(images, epsilon, gradient)

            optimizer.zero_grad()
            adv_outputs = model(adv_images.detach())
            adv_loss = criterion(adv_outputs, labels)

            adv_loss.backward()
            optimizer.step()

            total_loss += adv_loss.item()

        print(f"Defense Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -------------------------
# 7. RUN EVERYTHING
# -------------------------

print("\n--- TRAINING BASE MODEL ---")
train(model, train_loader, epochs=3)
baseline_acc = evaluate(model, test_loader)

print("\n--- ATTACKING MODEL (FGSM) ---")
fgsm_acc = evaluate_fgsm(model, test_loader, epsilon=0.25)

print("\n--- DEFENDING MODEL ---")
adversarial_train(model, train_loader, epochs=2, epsilon=0.25)

print("\n--- EVALUATING DEFENDED MODEL ---")
defended_acc = evaluate(model, test_loader)
defended_fgsm_acc = evaluate_fgsm(model, test_loader, epsilon=0.25)

# -------------------------
# 8. RESULTS GRAPH
# -------------------------
labels = ['Baseline', 'FGSM Attack', 'Defended', 'Defended + FGSM']
values = [baseline_acc, fgsm_acc, defended_acc, defended_fgsm_acc]

plt.bar(labels, values)
plt.ylabel("Accuracy")
plt.title("Model Performance Under Attack and Defense")
plt.show()

# -------------------------
# 9. FINAL PRINTS
# -------------------------
print("\nFINAL RESULTS")
print("Baseline Accuracy:", baseline_acc)
print("FGSM Accuracy:", fgsm_acc)
print("Defended Accuracy:", defended_acc)
print("Defended FGSM Accuracy:", defended_fgsm_acc)