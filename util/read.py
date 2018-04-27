from scipy.io.wavfile import read
import numpy as np
from util.classes import AudioData

def read_audio(path_to_wav):
    data = read(path_to_wav)
    sample_rate = data[0]
    audio_data = data[1]
    if len(audio_data.shape) == 2:
        return AudioData(sample_rate, audio_data[:,0])
    else:
        return AudioData(sample_rate, audio_data)

def main():
    # Open audio file
    import sys
    if(len(sys.argv) < 2):
        print("Usage: python chop.py path_to_wav_file")
        exit()
    path_to_file = sys.argv[1]
    read_audio(path_to_file)

if __name__ == '__main__':
    main()