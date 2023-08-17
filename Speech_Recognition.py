from __future__ import print_function
from tqdm import tqdm
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.model_selection import train_test_split
import tkinter.filedialog as ft
import random

testing_samples = 'up bird dog eight right zero left house happy no'.split()
num_classes = len(testing_samples)
input_shape = (177, 98, 1)


def create_training_data():
    path = r"E:\Dataset"
    training_data = []
    for sample in (os.listdir(path)):
        try:
            if sample in testing_samples:
                subpath = None
                subpath = (path + "/" + sample).replace('\\', '/')
                for i in tqdm(os.listdir(subpath)):
                    act_path = subpath + "/" + i
                    training_data.append([act_path, testing_samples.index(sample)])
        except Exception as e:
            pass
    return training_data


def process_wav_file(x, threshold_freq=5500, eps=1e-10):
    # Read wav file to array
    _, wav = wavfile.read(x)
    # Normalize
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    # Sample rate
    L = 16000
    # If longer then randomly truncate
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i + L)]
        # If shorter then randomly add silence
    elif len(wav) < L:
        rem_len = L - len(wav)
        silence_part = np.random.randint(-100, 100, 16000).astype(np.float32) / np.iinfo(np.int16).max
        j = np.random.randint(0, rem_len)
        silence_part_left = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    # Create spectrogram using discrete FFT (change basis to frequencies)
    freqs, times, spec = stft(wav, L, nperseg=400, noverlap=240, nfft=512, padded=False, boundary=None)
    # Cut high frequencies
    if threshold_freq is not None:
        spec = spec[freqs <= threshold_freq, :]
    # Log spectrogram
    amp = np.log(np.abs(spec) + eps)
    return np.expand_dims(amp, axis=2)


def extract_features(x_data):
    x_batch = []
    for i in tqdm(range(len(x_data))):
        x_batch.append(process_wav_file(x_data[i]))
    x_batch = np.array(x_batch)
    return x_batch


training_data = create_training_data()
random.shuffle(training_data)
train, validate = train_test_split(training_data, train_size=0.8)

x_train = []
y_train = []
x_validate = []
y_validate = []

for path, labels in train:
    x_train.append(path)
    y_train.append(labels)

for path, labels in validate:
    x_validate.append(path)
    y_validate.append(labels)

x_train = extract_features(x_train)
y_train = np.array(y_train)
x_validate = extract_features(x_validate)
y_validate = np.array(y_validate)

model = Sequential()

model.add((Conv2D(32, kernel_size=(3, 3),
                          activation='relu',
                          input_shape=input_shape)))
model.add((MaxPooling2D(pool_size=(3, 3))))

model.add((Conv2D(32, (3, 3), activation='relu')))
model.add((MaxPooling2D(pool_size=(3, 3))))

model.add((Conv2D(32, (3, 3), activation='relu')))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=15,
          verbose=1,
          validation_data=(x_validate, y_validate))

# serialize model to JSON
model_json = model.to_json()
with open("model9.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h9")


def test_model(model):
    dir = ft.askopenfilename(filetypes=(("Template files", "*.wav"), ("All files", "*")))
    try:
        test_data = []
        test_data.append(process_wav_file(dir))
        test_data = np.array(test_data)
        pred = model.predict(test_data)
        print('Predicted Result: ', testing_samples[pred.argmax()])
    except Exception as e:
        pass


# Load Model
json_file = open('model9.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h9")

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

while 1 == 1:
    test_model(loaded_model)



