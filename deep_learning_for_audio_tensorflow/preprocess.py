import os
import librosa


DATASET_PATH = 'Data/genres_original'
JSON_PATH = 'data.json'

SAMPLE_RATE = 22050

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    # dictionary to store data
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': []
    }
    
    # loop em todos os generos
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Ter certeza que não estamos no nível root
        if dirpath is not dataset_path:
            
            # save the semantic label
            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data['mapping'].append(semantic_label)
            
            # process files for a specific genre
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment* s
                    finish_sample = start_sample + num_samples_per_segment
                
            