import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo de audio
audio_path = './nuevo.wav'
y, sr = librosa.load(audio_path, sr=16000)

# Generar el espectrograma
n_fft = 2048  # Número de puntos FFT
hop_length = 512  # Salto entre ventanas

# Especificar el rango de frecuencias
min_freq = 50
max_freq = 8000

# Calcular el STFT (Short-Time Fourier Transform)
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

# Convertir amplitudes a decibelios
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Obtener las frecuencias correspondientes a las filas del espectrograma
frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# Filtrar el espectrograma para mantener solo las frecuencias deseadas
S_db_filtered = S_db[(frequencies >= min_freq) & (frequencies <= max_freq), :]

# Obtener las frecuencias filtradas
frequencies_filtered = frequencies[(frequencies >= min_freq) & (frequencies <= max_freq)]

# Visualizar el espectrograma
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db_filtered, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma ({} Hz - {} Hz)'.format(min_freq, max_freq))
plt.show()