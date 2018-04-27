import multiprocessing
import os
import sys
import time
from functools import partial
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy
from tqdm import tqdm

import pyaudio
from util.classes import (AudioData, amplitude_envelope, fourier,
                          sample_similarity, visual_similarity)
from util.read import read_audio

SLICE_SIZE = 44100 # Size, in samples, of the slices of audio
CACHE_ANALYSIS = True # If true (sorry, I mean True), save analysis data and use it to speed up processing

def main():
    # Get all files from audio folder and convert to AudioData
    
    dir = "audio"

    user_sample = False

    # This is exactly what you think it is
    if len(sys.argv) > 1: 
        if isfile(sys.argv[1]): # Passed in a sample to use instead of random
            user_sample = read_audio(sys.argv[1]).split(SLICE_SIZE)[0]
            if len(sys.argv) > 2:
                dir = sys.argv[2]
                print("Looking for files in directory:", dir)
        else: # Passed in a directory 
            dir = sys.argv[1]
            print("Looking for files in directory:", dir)
            if len(sys.argv) > 2:
                user_sample = read_audio(sys.argv[2]).split(SLICE_SIZE)[0]


    print("Loading data...")
    audios = []
    flattened_data = []
    for root, directories, filenames in os.walk(dir):
        for fname in filenames:
            if fname[-4:] == ".wav":
                audios.append(read_audio(join(root, fname)))
                print(join(root, fname))
            if fname[-4:] == ".mp3":
                pass # TODO
                
    if len(audios) == 0:
        print("Directory was empty. Exiting...")
        exit()


    # Split all audio files and extend flattened_data
    for a in audios:
        flattened_data.extend(a.split(SLICE_SIZE))
    flattened_data = np.array(flattened_data)

    # shuffle data
    np.random.shuffle(flattened_data)
    
    if user_sample is False:
        # pick a random sample from flattened_data
        print("Picking a random sample to find")
        random_chunk = flattened_data[0]
    else:
        # use the user_sample
        print("Using the user-provided sample")
        random_chunk = user_sample
    
    random_chunk_fft = fourier(random_chunk)
    random_chunk_amp = amplitude_envelope(random_chunk)
    
    # Find the most similar sample from flattened_data (that's not the random one)
    print("Starting to generate scores...")
    start = time.clock()
    pool = multiprocessing.Pool(1)
    pa = partial(sample_similarity, amp=random_chunk, ft=random_chunk_fft, debug=False)
    score_tuples = np.array(list(map(pa, flattened_data)))
    # score_tuples = np.array([sample_similarity(x, random_chunk_amp, random_chunk_fft, False) for x in flattened_data])
    amplitude_scores = score_tuples[:,0]
    fourier_scores = score_tuples[:,1]
    scores = amplitude_scores + 2*fourier_scores
    sorted_scores = np.argpartition(scores, 2)
    best_index = 0
    while scores[sorted_scores[best_index]] == 0: # Probably just the same sample
        best_index += 1
    end = time.clock()
    print("It took", end-start, "to do", SLICE_SIZE/44100, ". Ratio:", (SLICE_SIZE/44100)/(end-start))

    best = flattened_data[sorted_scores[best_index]]
    worst = flattened_data[np.argmax(scores)]

    # write the random sample and the most similar to disk
    scipy.io.wavfile.write("random.wav", 44100, np.asarray(random_chunk, dtype=np.int16))
    scipy.io.wavfile.write("best.wav", 44100, np.asarray(best, dtype=np.int16))
    scipy.io.wavfile.write("worst.wav", 44100, np.asarray(worst, dtype=np.int16))
    visual_similarity(random_chunk, best)

    # visual_similarity(random_chunk, worst)
    sample_similarity(best, random_chunk_amp, random_chunk_fft, True)
    sample_similarity(worst, random_chunk_amp, random_chunk_fft, True)
    p = pyaudio.PyAudio()

    print("Score: ", scores[sorted_scores[best_index]])

    # Play the samples out loud
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)
    stream.write(np.asarray(random_chunk, dtype=np.int16).copy(order='C'))
    stream.write(np.zeros(20000))
    stream.write(np.asarray(best, dtype=np.int16).copy(order='C'))
    stream.stop_stream()
    stream.close()
    p.terminate()



if __name__ == '__main__':
    main()
