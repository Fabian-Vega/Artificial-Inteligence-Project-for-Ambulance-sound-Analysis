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
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
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
