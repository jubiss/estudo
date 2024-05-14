import torch
import torchaudio


from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def predict(model, input, target, class_mapping):
    # Eval faz evaluation, dropout e outros layers úteis para o treino são removidos
    model.eval()
    # avalia o modelo sem fazer gradient descent
    with torch.no_grad():
        # Gera um tensor (Número de inputs, número de classes) Ex (1,10)
        # Esse tensor tem as probabilidades para cada classe
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet.pth")
    cnn.load_state_dict(state_dict)
    
    # load urban sound dataset validation dataset
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
                            device="cpu")

    # usd_sample = Subset(usd, np.arange(num_train_samples))
    
    # get a sample from the validation dataset for inference
    input, target = usd[0][0], usd[0][1]
    input.unsqueeze_(0)
    
    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted {predicted}, expected {expected}")