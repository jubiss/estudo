import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
# Normaliza todos os valores entre 0 e 1
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
class FeedFowardNet(nn.Module):    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    ) 
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()        
    ) 
    return train_data, validation_data

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"loss: {loss.item()}")
        
        
def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {(i+1)/epochs}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-"*15)
    print("Training is done")


if __name__ == '__main__':
    #Downlad MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")
    
    # Create a dataloader
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    
    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feed_foward_net = FeedFowardNet().to(device)
    
    # instantiate loss function + optimzer
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_foward_net.parameters(),
                                 lr=LEARNING_RATE)
    
    # Train model
    train(model=feed_foward_net, data_loader=train_data_loader, loss_fn=loss_fn, optimizer=optimizer, 
          device=device, epochs=EPOCHS)
    
    torch.save(feed_foward_net.state_dict(), "feedfowardnet.pth")
    print("Model trained and stored at feedfowardnet.pth")