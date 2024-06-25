import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import librosa

def load_wav_16k_mono(filename_tensor):
    """
    Load a WAV file from a TensorFlow tensor, convert it to a float tensor, and resample to 16 kHz single-channel audio.
    """
    # Convert the tensor to a NumPy string
    filename = filename_tensor.numpy().decode('utf-8')
    
    # Load the WAV file using librosa
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)  # `sr=16000` resamples to 16 kHz

    # Convert the numpy array to a TensorFlow tensor
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)

    return wav

# Wrap the load_wav_16k_mono function to make it compatible with TensorFlow's `tf.data.Dataset.map`
def load_wav_16k_mono_wrapper(filename_tensor):
    return tf.py_function(load_wav_16k_mono, [filename_tensor], tf.float32)


POS = os.path.join('data', 'sirens_wav') #Concatena la direccion de la carpeta con los wav de las sirenas
NEG = os.path.join( 'data', 'unheard_wav') #Concatena la direccion de la carpeta con los wav de las no sirenas


pos_files = os.listdir(POS) #Lista los archivos de la carpeta de sirenas
neg_files = os.listdir(NEG) #Lista los archivos de la carpeta de no sirenas

# Se crea un dataset de tensor flow con los archivos de sirenas y no sirenas
pos = tf.data.Dataset.list_files(POS + '/*_*.wav') # list_files agrega al dataset todos los archivos dado unos parametros
neg = tf.data.Dataset.list_files(NEG + '/*_*.wav')


#Se agrega una etiqueta a cada archivo, 1 para sirenas y 0 para no sirenas
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos))))) 
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives) # Une todos los datos en un mismo dataset

# Saca el promedio de la longitud de los audios
lengths = []
for file in os.listdir(os.path.join('data', 'unheard_wav')):
    file_path = os.path.join('data', 'unheard_wav', file)
    tensor_wave = load_wav_16k_mono(tf.convert_to_tensor(file_path))
    lengths.append(len(tensor_wave))
    
# Calcula el promedio, minimo y maximo de la longitud de los audios
tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)


def preprocess(file_path, label): 
    # Use the wrapper function to load the WAV file
    wav = load_wav_16k_mono_wrapper(file_path)
    wav = wav[:48000]  # Trim or pad the audio to 3 seconds (48,000 samples at 16 kHz)
    zero_padding = tf.zeros([48000 - tf.shape(wav)[0]], dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.reshape(spectrogram, (1491, 257))  # Adjust shape as per your data
    spectrogram = tf.expand_dims(spectrogram, axis=2)  # Add channel dimension
    return spectrogram, label



# 5.2 Test out the function and viz spectrogram
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
#plt.show()


# Creating Training and Testing Partisions
# 6.1 create a tensorflow data pipeline
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size = 10000)
data = data.batch(16)
data = data.prefetch(8)

# 6.2 Split into Training and Testing Partitions
train = data.take(36)
test = data.skip(36).take(15)

# 6.3 Test One Batch
samples, labels = train.as_numpy_iterator().next()
samples.shape


# Build Deep Learning Model
# 7.1 Load Tensorflow Dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout,BatchNormalization


# 7.2 Build Sequential Model, Compile and View Summary
model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu', input_shape=(1491,257,1)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.summary()

# Fit Model, View Loss and KPl Plots
history = model.fit(train, epochs=15, validation_data=test)

model.evaluate(test)

model.save('model.h5')

plt.title('Loss')
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'b')
plt.show()

# Make Prediction on single clip
# 8.1 get one batch and make prediction
xtest,ytest = test.as_numpy_iterator().next()
yhat = model.predict(xtest)

# Convert Logits to classes
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

from pydub import AudioSegment
import numpy as np

# Build Forest Parsing Function
# 9.1 load up mp3s
def load_mp3_16k_mono(filename):
    """
    Load an MP3 file, convert it to a float tensor, and resample to 16 kHz single-channel audio.
    """
    # Load the MP3 file using librosa
    # librosa.load returns the audio time series as a numpy array and the sampling rate
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)  # `sr=16000` resamples to 16 kHz

    # The `mono=True` argument ensures the audio is converted to a single channel (mono)
    # Resampling is automatically handled by librosa.load with the `sr` parameter

    return wav


mp3 = os.path.join('data','testingAudios','2024-06-24-21-40-03.mp3')
wav = load_mp3_16k_mono(mp3)
audio_slice = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
sample, index = audio_slice.as_numpy_iterator().next()

# 9.2 Build Function to convert clips into windowed spectrogram
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# 9.3 convert longer clips into windows and make predictions
audio_slice = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slice = audio_slice.map(preprocess_mp3)
audio_slice = audio_slice.batch(64)
yhat = model.predict(audio_slice)
yhat  = [ 1 if prediction > 0.5 else 0 for prediction in yhat]

# 9.4 Group Consenutive Detectons
from itertools import groupby
yhat = [ key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()
calls

# 10 Make Prediction
# 10.1 Loop over all recordings and make prediction
results = {}
for file in os.listdir(os.path.join('data','testingAudios')):
    FILEPATH = os.path.join('data','testingAudios',file)
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    yhat = model.predict(audio_slices)
    results[file] = yhat
    
    
# Convert predictions into classes
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
# class_preds
# 10.3 Group Consective Detections
postprocessd = {}
for file, scores in class_preds.items():
    postprocessd[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessd

# 11 Export Results
import csv
with open('results.csv','w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recoding', 'ambulances'])
    for key, value in postprocessd.items():
        writer.writerow([key, value])