import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio


POS = os.path.join('data', 'sirens_wav')
NEG = os.path.join( 'data', 'unheard_wav')


pos_files = os.listdir(POS)
neg_files = os.listdir(NEG)

#print("Positive files:", pos_files)
#print("Negative files:", neg_files)

# 3.2 Create Tensorflow Datasets
pos = tf.data.Dataset.list_files(os.path.join(POS + '\*-*.wav'))
neg = tf.data.Dataset.list_files(os.path.join(NEG + '\*-*.wav'))

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
