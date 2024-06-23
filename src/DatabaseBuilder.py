import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    #wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    
    return wav



POS = os.path.join('data', 'sirens_wav')
NEG = os.path.join( 'data', 'unheard_wav')


pos_files = os.listdir(POS)
neg_files = os.listdir(NEG)

#print("Positive files:", pos_files)
#print("Negative files:", neg_files)

# 3.2 Create Tensorflow Datasets
pos = tf.data.Dataset.list_files(POS + '/*_*.wav')
neg = tf.data.Dataset.list_files(NEG + '/*_*.wav')

# 3.2 Add labels and combine positive and negative samples
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

# 4. Determine Average Length of a capucin call
# 4.1 calculate wave cycle length
lengths = []
for file in os.listdir(os.path.join('data', 'unheard_wav')):
    file_path = os.path.join('data', 'unheard_wav', file)
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
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization


# 7.2 Build Sequential Model, Compile and View Summary
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
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

