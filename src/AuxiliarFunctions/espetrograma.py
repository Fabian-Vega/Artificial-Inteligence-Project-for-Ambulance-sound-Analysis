import matplotlib.pyplot as plt
from scipy.io import wavfile

# Cargar archivo de audio
sample_rate, wave_data = wavfile.read('./data/sirens/siren_048.wav')

# Asegurar que la señal sea unidimensional (si es necesario)
if wave_data.ndim > 1:
    wave_data = wave_data[:, 0]  # Tomar solo el primer canal si hay múltiples canales

# Crear espectrograma
plt.figure(figsize=(10, 6))
plt.specgram(wave_data, Fs=sample_rate)
plt.title('Espectrograma del Audio')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.colorbar(label='Intensidad [dB]')
plt.show()