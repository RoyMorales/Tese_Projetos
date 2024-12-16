# Testing Pytoarch

from types import DynamicClassAttribute
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from torchvision.models  import vit_b_16, ViT_B_16_Weights

device = (
        "cuda" if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)

print(f"\nUsing {device} device\n")

#########################################################

def load_data():
    
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize CIFAR-10 (32x32) to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform,
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    print("\n")
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    print("\n")


    return train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optmizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optmizer.step()
        optmizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss:>8f}\n")

if __name__ == '__main__':
    batch_size = 128
    epochs = 50
    num_classes = 10

    torch.backends.cudnn.benchmark = True

    training_dataloader, test_dataloader = load_data()

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for t in range(epochs):
        print(f"Epoch {t+1}\n----------------------------")
        train(training_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    
    torch.save(model.state_dict(), 'VGG.pth')
    print("Saved PyTorch Model State to VGG.pth")
