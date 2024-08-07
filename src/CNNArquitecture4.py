import os
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, MaxPooling2D # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
import time
start_time = time.time()  # Start the timer
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

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


POS = os.path.join('data', 'sirens') #Concatena la direccion de la carpeta con los wav de las sirenas
NEG = os.path.join( 'data', 'unheard') #Concatena la direccion de la carpeta con los wav de las no sirenas


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
for file in os.listdir(os.path.join('data', 'unheard')):
    file_path = os.path.join('data', 'unheard', file)
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


# 7.2 Build Sequential Model, Compile and View Summary

# Define layer names
# conv_layer_1 = Conv2D(32, (3, 3), activation='relu', input_shape=(1491, 257, 1), name='Conv_Layer_1')
# max_pooling_layer_2 = MaxPooling2D(pool_size=(2, 2), name='Max_Pooling_Layer_2')
# batch_norm_layer_2 = BatchNormalization(name='BatchNorm_Layer_2')
# batch_norm_layer_1 = BatchNormalization(name='BatchNorm_Layer_1')

max_pooling_layer_1 = MaxPooling2D(pool_size=(2, 2),input_shape=(1491, 257, 1), name='Max_Pooling_Layer_1')
conv_layer_1 = Conv2D(32, (3, 3), activation='relu', name='Conv_Layer_1')
conv_layer_2 = Conv2D(64, (3, 3), activation='relu', name='Conv_Layer_2')
flatten_layer = Flatten(name='Flatten_Layer')
output_layer = Dense(1, activation='sigmoid', name='Output_Layer')

# Create a sequential model
model = Sequential(name='Siren_Detection_Architecture_4')

# Add layers to the model
model.add(max_pooling_layer_1)
model.add(conv_layer_1)
model.add(conv_layer_2)
model.add(flatten_layer)
model.add(output_layer)



# model.add(max_pooling_layer_1)
# model.add(batch_norm_layer_1)
# model.add(max_pooling_layer_2)
# model.add(batch_norm_layer_2)


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
model.summary()

# Fit Model, View Loss and KPl Plots
history = model.fit(train, epochs=5, validation_data=test)

model.evaluate(test)

end_time = time.time()  # End the timer

# Calculate the execution time
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
end_time = time.time()  # End the timer
execution_time = end_time - start_time
print(f"Program executed in: {execution_time} seconds")
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

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