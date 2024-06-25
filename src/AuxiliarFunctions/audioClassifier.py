import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

#Plotea los waves
# # 2. Build Data Loading Function
# # 2.1 Define Paths To Files
# SIREN_FILE = os.path.join('data', '..\data\sirens','XC114131-0.wav')
# NOT_SIREN_FILE = os.path.join('data', '..\data\sirens', 'afternoon-birds-song-in-forest-0.wav')

# # 2.1 Build Dataloading Function
# def load_wav_16k_mono(filename):
#     # Load encoded wav file
#     file_contents = tf.io.read_file(filename)
#     # Decode wav (tensors by channels) 
#     wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
#     # Removes trailing axis
#     wav = tf.squeeze(wav, axis=-1)
#     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#     # Goes from 44100Hz to 16000hz - amplitude of the audio signal
#     wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
#     return wav

# # 2.2 Plot Wave

# wave = load_wav_16k_mono(SIREN_FILE)
# nwave = load_wav_16k_mono(NOT_SIREN_FILE)

# plt.plot(wave)
# plt.plot(nwave)
# plt.show()

# Create Tensorflow Dataset
# 3.1 Define paths to positive and negative data

POS = os.path.join('..', 'data', 'sirens')
NEG = os.path.join('..', 'data', 'unheard')


pos_files = os.listdir(POS)
neg_files = os.listdir(NEG)

print("Positive files:", pos_files)
print("Negative files:", neg_files)


# 3.2 Create Tensorflow Datasets
pos = tf.data.Dataset.list_files(POS + '/*-*.wav')
neg = tf.data.Dataset.list_files(NEG + '/*-*.wav')

# 3.2 Add labels and combine positive and negative samples
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

# 4. Determine Average Length of a capucin call
# 4.1 calculate wave cycle length
lengths = []
for file in os.listdir(os.path.join('data','/kaggle/input/z-by-hp-unlocked-challenge-3-signal-processing/Parsed_Not_Capuchinbird_Clips')):
    file_path = os.path.join('data','/kaggle/input/z-by-hp-unlocked-challenge-3-signal-processing/Parsed_Not_Capuchinbird_Clips', file)
#     print("Current file path:", file_path)
    tensor_wave = load_wav_16k_mono(file_path)
    lengths.append(len(tensor_wave))
    
    
# 4.2 calculate mean, min and max
tf.math.reduce_mean(lengths)

tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

# Build Preprocessing Function
# 5.1 Build Preprocessing Function
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


# 5.2 Test out the function and viz spectrogram
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()


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
# model.add(Dense(128, activation='relu'))
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

# Build Forest Parsing Function
# 9.1 load up mp3s
def load_mp3_16k_mono(filename):
    """ Load an MP3 file, convert it to a float tensor, and resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    
    # Convert to tensor and combine channels
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    # Resample to 16 KHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    
    return wav
mp3 = os.path.join('data','/kaggle/input/z-by-hp-unlocked-challenge-3-signal-processing/Forest Recordings','/kaggle/input/z-by-hp-unlocked-challenge-3-signal-processing/Forest Recordings/recording_00.mp3')
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
for file in os.listdir(os.path.join('data','/kaggle/input/z-by-hp-unlocked-challenge-3-signal-processing/Forest Recordings')):
    FILEPATH = os.path.join('data','/kaggle/input/z-by-hp-unlocked-challenge-3-signal-processing/Forest Recordings',file)
    
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
    writer.writerow(['recoding', 'capuchin_calls'])
    for key, value in postprocessd.items():
        writer.writerow([key, value])