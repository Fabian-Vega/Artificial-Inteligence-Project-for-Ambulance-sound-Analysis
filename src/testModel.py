import os
import tensorflow as tf
import librosa
from itertools import groupby

def load_mp3_16k_mono(filename):
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000 - tf.shape(sample)[0]], dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

if __name__ == "__main__":
    model = tf.keras.models.load_model('model.h5')
    results = {}

    for file in os.listdir(os.path.join('data', 'testingAudios')):
        FILEPATH = os.path.join('data', 'testingAudios', file)
        
        wav = load_mp3_16k_mono(FILEPATH)
        audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=24000, batch_size=1)
        audio_slices = audio_slices.map(preprocess_mp3)
        audio_slices = audio_slices.batch(64)
        yhat = model.predict(audio_slices)
        results[file] = yhat

    class_preds = {}
    for file, logits in results.items():
        class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]

    postprocessd = {}
    for file, scores in class_preds.items():
        postprocessd[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()

    import csv
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['recording', 'ambulances'])
        for key, value in postprocessd.items():
            writer.writerow([key, value])
