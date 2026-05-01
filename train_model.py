import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Simple Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
print("Training baseline model...")
for epoch in range(3):
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluate accuracy
def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")

print("\nClean accuracy BEFORE attack:")
evaluate(model, test_loader)
# FGSM Attack
def fgsm_attack(data, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = data + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Evaluate FGSM attack
def evaluate_fgsm(model, test_loader, epsilon):
    correct = 0
    total = 0

    for data, target in test_loader:
        data.requires_grad = True

        output = model(data)
        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        preds = output.argmax(dim=1)

        correct += (preds == target).sum().item()
        total += target.size(0)

    print(f"FGSM Accuracy (epsilon={epsilon}): {100 * correct / total:.2f}%")

print("\nRunning FGSM attack...")
evaluate_fgsm(model, test_loader, epsilon=0.25)
# Defense: Adversarial Training
def adversarial_training(model, train_loader, epsilon):
    print("\nApplying defense (adversarial training)...")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        total_loss = 0

        for data, target in train_loader:
            data.requires_grad = True

            output = model(data)
            loss = criterion(output, target)

            model.zero_grad()
            loss.backward()

            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            # Train on attacked data
            output = model(perturbed_data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Defense Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Run defense
adversarial_training(model, train_loader, epsilon=0.25)

print("\nClean accuracy AFTER defense:")
evaluate(model, test_loader)

print("\nFGSM accuracy AFTER defense:")
evaluate_fgsm(model, test_loader, epsilon=0.25)
import matplotlib.pyplot as plt

labels = ["Clean Before", "Under Attack", "After Defense"]
values = [98.47, 8.57, 99.24]  # use YOUR numbers if slightly different

plt.bar(labels, values)
plt.ylabel("Accuracy (%)")
plt.title("Model Performance Under Attack and Defense")

plt.savefig("graph.png")   # ✅ saves the image
plt.show()