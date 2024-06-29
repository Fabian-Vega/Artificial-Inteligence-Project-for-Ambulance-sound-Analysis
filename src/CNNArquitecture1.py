import os
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore



def load_wav_16k_mono(filename_tensor):
    """
    Load a WAV file from a TensorFlow tensor, convert it to a float tensor, and resample to 16 kHz single-channel audio.
    """
    # Convert the tensor to a NumPy string
    filename = filename_tensor.numpy().decode('utf-8')
    
    # Load the WAV file using librosa
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)  # sr=16000 resamples to 16 kHz

    # Convert the numpy array to a TensorFlow tensor
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)

    return wav

# Wrap the load_wav_16k_mono function to make it compatible with TensorFlow's tf.data.Dataset.map
def load_wav_16k_mono_wrapper(filename_tensor):
    return tf.py_function(load_wav_16k_mono, [filename_tensor], tf.float32)




POS = os.path.join('data', 'sirens') 
NEG = os.path.join('data', 'unheard')

# VAL_POS = os.path.join('data', 'sirens_validation')
# VAL_NEG = os.path.join('data', 'unheard_validation')

pos_files = os.listdir(POS) 
neg_files = os.listdir(NEG)

# val_pos_files = os.listdir(VAL_POS)
# val_neg_files = os.listdir(VAL_NEG)

# Create TensorFlow datasets for training and validation
pos = tf.data.Dataset.list_files(POS + '/*_*.wav') 
neg = tf.data.Dataset.list_files(NEG + '/*_*.wav')

# val_pos = tf.data.Dataset.list_files(VAL_POS + '/*_*.wav')
# val_neg = tf.data.Dataset.list_files(VAL_NEG + '/*_*.wav')

# Add labels to the datasets
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos_files))))) 
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg_files)))))

# val_positives = tf.data.Dataset.zip((val_pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(val_pos_files)))))
# val_negatives = tf.data.Dataset.zip((val_neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(val_neg_files)))))

# Combine positive and negative datasets for training and validation
train_data = positives.concatenate(negatives)
# val_data = val_positives.concatenate(val_negatives)



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


# Preprocess and batch the dataset
train_data = train_data.map(preprocess)
train_data = train_data.cache()
train_data = train_data.shuffle(buffer_size=10000)
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# val_data = val_data.map(preprocess)
# val_data = val_data.cache()
# val_data = val_data.batch(16)
# val_data = val_data.prefetch(8)

# Print total number of samples
print(f"Total number of training samples: {len(pos_files) + len(neg_files)}")
# print(f"Total number of validation samples: {len(val_pos_files) + len(val_neg_files)}")

# train = data.take(36)
# test = data.skip(36).take(15)

# # 6.3 Test One Batch
# samples, labels = train.as_numpy_iterator().next()
# samples.shape


# Build Deep Learning Model
# 7.1 Load Tensorflow Dependencies



# Define layers and their names
conv_layer_1 = Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1), name='Conv_Layer_1')
conv_layer_2 = Conv2D(16, (3, 3), activation='relu', name='Conv_Layer_2')
flatten_layer = Flatten(name='Flatten_Layer')
output_layer = Dense(1, activation='sigmoid', name='Output_Layer')

# Create a sequential model
model = Sequential(name='Siren_Detection_Architecture_1')

# Add layers to the model
model.add(conv_layer_1)
model.add(conv_layer_2)
model.add(flatten_layer)
model.add(output_layer)

# Print model summary
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.summary()

# Fit Model, View Loss and KPl Plots
history = model.fit(train_data, epochs=5)

# model.evaluate(val_data)

model.save('model.h5')


# Plot the training history
output_dir = 'output_graphs'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], 'r', label='Pérdida de Entrenamiento')
# plt.plot(history.history['val_loss'], 'b', label='Pérdida de Validación')
plt.title('Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history['recall'], 'r', label='Recall de Entrenamiento')
# plt.plot(history.history['val_recall'], 'b', label='Recall de Validación')
plt.title('Recall')
plt.xlabel('Épocas')
plt.ylabel('Recall')
plt.legend()
plt.savefig(os.path.join(output_dir, 'recall_plot.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history['precision'], 'r', label='Precisión de Entrenamiento')
# plt.plot(history.history['val_precision'], 'b', label='Precisión de Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.savefig(os.path.join(output_dir, 'precision_plot.png'))
plt.close()