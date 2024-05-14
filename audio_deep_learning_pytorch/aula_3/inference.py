import torch
from aula_5.train import FeedFowardNet, download_mnist_datasets

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
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
    feed_foward_net = FeedFowardNet()
    state_dict = torch.load("feedfowardnet.pth")
    feed_foward_net.load_state_dict(state_dict)
    
    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()
    
    # get a sample from the validation dataset for inference
    input, target = validation_data[5][0], validation_data[5][1]
    
    # make an inference
    predicted, expected = predict(feed_foward_net, input, target,
                                  class_mapping)
    
    print(f"Predicted {predicted}, expected {expected}")