import os
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#This function loads audio files, resamples them to 16 kHz, 
# and converts them to mono (single-channel audio). It uses librosa for 
# loading and resampling and TensorFlow for handling tensor conversions.
def load_wav_16k_mono(filename_tensor):
    filename = filename_tensor.numpy().decode('utf-8')
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    return wav

def load_wav_16k_mono_wrapper(filename_tensor):
    return tf.py_function(load_wav_16k_mono, [filename_tensor], tf.float32)

#This function converts the audio file to a Short-Time Fourier Transform (STFT) 
# spectrogram, which is a 2D representation of the audio signal, with one dimension 
# representing time and the other frequency. This spectrogram is the input to the CNN model.
def preprocess(file_path, label):
    wav = load_wav_16k_mono_wrapper(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000 - tf.shape(wav)[0]], dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.reshape(spectrogram, (1491, 257))
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

def create_datasets(pos_dir, neg_dir):
    pos_files = os.listdir(pos_dir)
    neg_files = os.listdir(neg_dir)

    pos = tf.data.Dataset.list_files(pos_dir + '/*_*.wav')
    neg = tf.data.Dataset.list_files(neg_dir + '/*_*.wav')

    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))

    data = positives.concatenate(negatives)

    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)
    data = data.batch(16)
    data = data.prefetch(8)

    train = data.take(36)
    test = data.skip(36).take(15)

    return train, test

#This function constructs a simple CNN using TensorFlow's Keras API. 
# The model includes convolutional layers that are crucial for learning patterns 
# in the spectrogram data.
def build_model():
    #Sequential groups a linear stack of layers into a Model.
    model = Sequential()
    # Conv2D(Number of filters, (Size of each filter in pixels), The Rectified Linear Unit (ReLU) 
    # activation function introduces non-linearity, allowing the network to learn complex patterns,
    # input_shape=(The shape of the input data. For spectrograms, this means 1491 time steps, 257 frequency bins,
    # and 1 channel.)) 
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    return model

def plot_loss(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'b')
    plt.show()

if __name__ == "__main__":
    POS = os.path.join('data', 'sirens_wav')
    NEG = os.path.join('data', 'unheard_wav')

    train, test = create_datasets(POS, NEG)
    
    model = build_model()
    model.summary()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train, epochs=15, validation_data=test, callbacks=[early_stopping])

    model.evaluate(test)
    model.save('model.h5')

    plot_loss(history)
