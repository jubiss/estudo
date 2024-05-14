import torch
import torchaudio
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

ANNOTATIONS_FILE = r'metadata/UrbanSound8K.csv'
AUDIO_DIR = r'audio/'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
num_train_samples = 3000

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader

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

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    # instatiating our dataset objects and create data loader
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
        )
    
    usd = UrbanSoundDataset(annotations_file=ANNOTATIONS_FILE,
                            audio_dir=AUDIO_DIR,
                            transformation=mel_spectogram,
                            target_sample_rate=SAMPLE_RATE,
                            num_samples=NUM_SAMPLES,
                            device=device)

    usd_sample = Subset(usd, np.arange(num_train_samples))
    # Create a dataloader
    train_data_loader = DataLoader(usd_sample, batch_size=BATCH_SIZE)
    
    cnn = CNNNetwork().to(device)
    print(cnn)
    
    # instantiate loss function + optimzer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)
    
    # Train model
    train(model=cnn, data_loader=train_data_loader, loss_fn=loss_fn, optimizer=optimizer, 
          device=device, epochs=EPOCHS)
    
    torch.save(cnn.state_dict(), "feedfowardnet.pth")
    print("Model trained and stored at feedfowardnet.pth")