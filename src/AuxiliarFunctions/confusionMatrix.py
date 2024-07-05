import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Cargar datos desde el archivo CSV (reemplaza 'results4.csv' con tu archivo)
df = pd.read_csv('Arquitecture2/results2.csv')

# Definir etiquetas verdaderas y predicciones
y_true = df['ambulances']  # Etiquetas verdaderas
y_pred = [1 if filename.startswith('sirens') else 0 for filename in df['recording']]  # Predicciones

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Reordenar la matriz de confusión según el formato deseado
conf_matrix_reordered = [
    [conf_matrix[1, 1], conf_matrix[0, 1]],
    [conf_matrix[1, 0], conf_matrix[0, 0]]
]

# Mostrar la matriz de confusión como gráfico
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_reordered, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()

classes = ['Positivo (1)', 'Negativo (0)']
tick_marks = [0, 1]

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

thresh = max(map(max, conf_matrix_reordered)) / 2.
for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    plt.text(j, i, format(conf_matrix_reordered[i][j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_reordered[i][j] > thresh else "black")

plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Predicción')
plt.tight_layout()
plt.show()
