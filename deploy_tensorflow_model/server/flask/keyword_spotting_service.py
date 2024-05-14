import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = 'model.h5'
NUM_SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    
    # Singleton classe que só pode ter um objeto

    model = None
    _mappings = [
        "train\\bed",
        "train\\bird",
        "train\\cat",
        "train\\dog",
        "train\\down",
        "train\\eight",
        "train\\five",
        "train\\four",
        "train\\go",
        "train\\happy",
        "train\\house",
        "train\\left",
        "train\\marvin",
        "train\\nine",
        "train\\no",
        "train\\off",
        "train\\on",
        "train\\one",
        "train\\right",
        "train\\seven",
        "train\\sheila",
        "train\\six",
        "train\\stop",
        "train\\three",
        "train\\tree",
        "train\\two",
        "train\\up",
        "train\\wow",
        "train\\yes",
        "train\\zero",
        "train\\_background_noise_"
    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path)

        # convert 2d MFCCs array into 4d array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction    
        predictions = self.model.predict(MFCCs) 
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        
        # load audio
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) >= NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]
        else:
            signal = np.pad(signal, pad_width=((NUM_SAMPLES_TO_CONSIDER - signal.shape[0])))

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service():
    # Implementação de singleton
    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == '__main__':
    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("data/test/brid.wav")
    # keyword2 = kss.predict("data/test/dog.wav")

    print(f"predicted keywords:  {keyword1}")