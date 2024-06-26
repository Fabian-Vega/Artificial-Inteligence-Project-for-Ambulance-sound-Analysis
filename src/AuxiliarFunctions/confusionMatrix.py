import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Leer el archivo CSV
data = pd.read_csv('./results/results15.csv')

# Crear las etiquetas verdaderas (y_true)
data['true_label'] = data['recording'].apply(lambda x: 0 if 'NOAmbulance' in x else 1)

# Procesar los valores de ambulances
data['predicted_label'] = data['ambulances'].apply(lambda x: 1 if x > 1 else x)

# Extraer las etiquetas verdaderas y las predicciones
y_true = data['true_label']
y_pred = data['predicted_label']

# Crear la matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Reorganizar la matriz de confusión para que TP esté arriba a la izquierda
reordered_cm = np.array([
    [cm[1, 1], cm[1, 0]],
    [cm[0, 1], cm[0, 0]]
])

# Crear un gráfico de matriz de confusión con etiquetas numéricas
disp = ConfusionMatrixDisplay(confusion_matrix=reordered_cm, display_labels=['positivo', 'negativo'])
disp.plot(cmap=plt.cm.Blues, colorbar=False)
plt.title('Matriz de Confusión con 15 épocas')
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.show()

# Imprimir los valores de la matriz de confusión reorganizada
print('Matriz de Confusión Reorganizada:\n', reordered_cm)

# Extraer valores TN, FP, FN, TP de la matriz reorganizada
tn, fp, fn, tp = reordered_cm.ravel()
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
print(f'True Positives (TP): {tp}')