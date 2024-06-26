import requests

URL = "http://13.58.245.48/predict"
TEST_AUDIO_FILE_PATH = "test/dog.wav" #"data/test/brid.wav"


if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, 'rb')
    values = {'file': (TEST_AUDIO_FILE_PATH, audio_file, 'audio/wav')}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")