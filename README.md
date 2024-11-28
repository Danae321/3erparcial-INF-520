# 3erparcial-INF-520

# Perceptrón Multicapa para Clasificación del Dataset Iris

## I. Modelado Manual

El objetivo es implementar el cálculo manual del Perceptrón Multicapa para clasificar si una flor corresponde a la especie *Setosa* (`1`) o no (`0`), usando los pesos y parámetros adecuados.
### Ejemplo con la Primera Muestra

- **Entrada:**  
  \[
  x = [5.1, 3.5, 1.4, 0.2]
  \]
- **Pesos iniciales:**  
  \[
  w = [0.5, -0.3, 0.8, -0.2], \, b = 0.1
  \]
  Resolviendo:
  \[
  z = 2.55 - 1.05 + 1.12 - 0.04 + 0.1 = 2.68
  \]

- **Activación:**
  \[
  a = \frac{1}{1 + e^{-2.68}} \approx 0.935
  \]

Este procedimiento se puede replicar para todas las muestras del conjunto de datos.

---

## II. Programa Automatizado

Utilizaremos Python y la biblioteca `scikit-learn` para entrenar un Perceptrón Multicapa. 

### Código Python

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Datos ajustados
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.7, 3.2, 1.4, 0.2],
    [7.7, 3.0, 6.1, 2.3],
    [6.8, 3.2, 5.9, 2.3],
    [4.7, 3.2, 1.4, 0.2],
    [6.4, 3.1, 5.5, 1.8],
    [5.0, 3.4, 1.5, 0.2],
    [4.6, 3.4, 1.4, 0.3],
])
y = np.array([1, 1, 0, 0, 1, 0, 1, 1])  # Etiquetas ajustadas

# Entrenar un Perceptrón Multicapa
clf = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', max_iter=1000, random_state=1)
clf.fit(X, y)

# Predicción
predicciones = clf.predict(X)
print("Predicciones:", predicciones)

# Visualización
for i, (real, pred) in enumerate(zip(y, predicciones)):
    print(f"Flor {i+1}: Real={real}, Predicción={pred}")

# Gráfico de clasificación (usando solo dos características para facilitar visualización)
plt.figure()
plt.scatter(X[y == 1][:, 2], X[y == 1][:, 3], label='Setosa (Real)', color='blue')
plt.scatter(X[y == 0][:, 2], X[y == 0][:, 3], label='No Setosa (Real)', color='red')
plt.title("Clasificación Binaria: Perceptrón Multicapa")
plt.xlabel("Largo de pétalo")
plt.ylabel("Ancho de pétalo")
plt.legend()
plt.show()
