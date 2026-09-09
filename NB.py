import numpy as np
import pandas as pd
import math as mt

from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayes(ClassifierMixin, BaseEstimator):
    def __init__(self, m=1.0):
        self.m = m                     # float: hiperparametro del m-estimador
        self.classes_ = None             # list[str]: clases posibles, ej ["E", "G", "P"]
        self.columnas = None           # list[str]: nombres de los atributos
        self.prior = {}                # dict[str, float]: prior[clase] = P(clase)
        self.cantidad_por_clase = {}   # dict[str, int]: filas de cada clase (el "n" de la formula)
        self.p_attr = {}               # dict[str, float]: p_attr[col] = 1 / (valores distintos)
        self.cond = {}                 # dict: cond[col][valor][clase] = P(col = valor | clase)

    # Entrenamiento 
    def fit(self, X, y):
        # X: DataFrame, una columna por atributo
        # y: Series, la clase de cada fila ("E" / "G" / "P")

        lista_clases = sorted(y.unique())
        lista_columnas = list(X.columns)
        cantidad_filas = len(y)

        self.classes_ = np.array(lista_clases)
        self.columnas = lista_columnas

        # Prior: probabilidad de cada clase, sin mirar los atributos
        #     P(clase) = filas de esa clase / filas totales
        for clase in lista_clases:
            filas_de_la_clase = (y == clase).sum()               # int
            self.cantidad_por_clase[clase] = filas_de_la_clase
            self.prior[clase] = filas_de_la_clase / cantidad_filas

        for nombre_col in lista_columnas:
            valores_posibles = X[nombre_col].unique()            # array de str
            p = 1.0 / len(valores_posibles)                      # float
            self.p_attr[nombre_col] = p
            self.cond[nombre_col] = {}

            for valor in valores_posibles:
                self.cond[nombre_col][valor] = {}

                for clase in lista_clases:
                    mascara_clase = (y == clase)                             # Series de True/False
                    mascara_clase_y_valor = mascara_clase & (X[nombre_col] == valor)

                    n = mascara_clase.sum()                                  # int
                    n_c = mascara_clase_y_valor.sum()                        # int

                    self.cond[nombre_col][valor][clase] = (n_c + self.m * p) / (n + self.m)

        return self

    def _score_fila(self, fila, clase):
        sumaLog = 0
        for col in self.columnas:
            valor = fila[col]
            if valor in self.cond[col]:
                prob = self.cond[col][valor][clase]
            else:
                prob = (0 + self.m * self.p_attr[col]) / (self.cantidad_por_clase[clase] + self.m)

            sumaLog += mt.log2(prob)
        res = sumaLog + mt.log2(self.prior[clase])
        return res

    def predict(self, X):
        resultados = []
        for _, fila in X.iterrows():
            scores = {clase: self._score_fila(fila, clase) for clase in self.classes_}
            resultado = max(scores, key=scores.get)   # la clave (clase) con score mas alto
            resultados.append(resultado)
        return np.array(resultados)

