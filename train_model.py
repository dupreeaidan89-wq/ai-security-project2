import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

# Neural Network
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

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
def train():
    model.train()
    for epoch in range(3):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")

# Test Accuracy
def test():
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += 1
    acc = correct / total
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

# FGSM Attack
def fgsm_attack(data, epsilon, gradient):
    sign = gradient.sign()
    perturbed = data + epsilon * sign
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed

# Test with Attack
def test_fgsm(epsilon):
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()

        perturbed_data = fgsm_attack(data, epsilon, data.grad.data)
        output = model(perturbed_data)

        pred = output.argmax(dim=1)
        if pred.item() == target.item():
            correct += 1

    acc = correct / len(test_loader)
    print(f"FGSM Accuracy (epsilon={epsilon}): {acc*100:.2f}%")
    return acc

# Defense: Adversarial Training
def adversarial_train(epsilon=0.2):
    model.train()
    for epoch in range(2):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True

            output = model(data)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()

            perturbed_data = fgsm_attack(data, epsilon, data.grad.data)

            optimizer.zero_grad()
            output = model(perturbed_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Adversarial Epoch {epoch+1} complete")

# Run everything
train()

print("\n--- Normal Accuracy ---")
test()

print("\n--- Under Attack ---")
test_fgsm(0.2)

print("\n--- Applying Defense ---")
adversarial_train()

print("\n--- Accuracy After Defense ---")
test()

print("\n--- Attack After Defense ---")
test_fgsm(0.2)
import matplotlib.pyplot as plt

accuracies = [96.9, 0.32, 45.5]
labels = ["Normal", "Under Attack", "After Defense"]

plt.plot(labels, accuracies, marker='o')
plt.title("Model Accuracy Under Attack and Defense")
plt.ylabel("Accuracy (%)")
plt.show()
# Second Graph: Accuracy vs Epsilon (Attack Strength)

epsilons = [0, 0.05, 0.1, 0.15, 0.2]
eps_accuracies = []

for eps in epsilons:
    acc = test_fgsm(eps)
    eps_accuracies.append(acc * 100)

plt.figure()
plt.plot(epsilons, eps_accuracies, marker='o')
plt.title("FGSM Attack Strength vs Accuracy")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()