import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Cargar datos desde el archivo CSV
df = pd.read_csv('results2.csv')

# Definir etiquetas verdaderas y predicciones
y_true = df['ambulances']  # Etiquetas verdaderas
y_pred = [1 if filename.startswith('sirens') else 0 for filename in df['recording']]  # Predicciones

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Mostrar la matriz de confusión como gráfico
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()

classes = ['Positvo (1)', 'Negativo (0)']
tick_marks = [0, 1]

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

thresh = conf_matrix.max() / 2.
for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Predicción')
plt.tight_layout()
plt.show()
