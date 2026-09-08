#Arbol de Decision ID3:
# 1. Defino los posibles selectores de atributo
# 2. Algoritmo ID3
    # 2.1 Si todos los ejemplos tienen el mismo valor etiqueto con ese valor
    # 2.2 Si no quedan mas atributos pongo una hoja 
    # 2.3 Seleccionar atributo con mayor score 
    # 2.4 En caso contrario etiqueto con un atributo disponible 
    # 2.5 Para cada valor posible del atributo etiquetado genero una rama etiquetada con ese valor 
    # 2.6 Si esa rama no tiene instacias etiqueto con el valor mas comun de ejemplos 
    # 2.7 Hago la recursion sacando el atributo etiquetado y solo con las instancias de ese valor

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

# 1.1 ImpurityReduction (Gini)
def gini(y):
    if len(y) == 0:
        return 0
    conteos = y.value_counts()
    probs = conteos / len(y)
    return 1 - np.sum(probs ** 2)

def impurityReduction(X, y, atributo):
    gini_padre = gini(y)
    n_total = len(y)
 
    gini_ponderado = 0
    for valor in X[atributo].unique():
        mascara = X[atributo] == valor
        subset_y = y[mascara]
        peso = len(subset_y) / n_total
        gini_ponderado += peso * gini(subset_y)
 
    return gini_padre - gini_ponderado


# 1.2 GainRatio (SplitInformation)
def split_information(X, atributo):
    n_total = len(X)
    conteos = X[atributo].value_counts()
    probs = conteos / n_total
    return -np.sum(probs * np.log2(probs))
 
def gain_ratio(X, y, atributo):
    ganancia = ganancia_informacion(X, y, atributo)
    si = split_information(X, atributo)
 
    if si == 0:
        return 0
 
    return ganancia / si


# 1.3 Ganancia (Entropy)
def entropia(y):
    if len(y) == 0:
        return 0
    conteos = y.value_counts()
    probs = conteos / len(y)
    return -np.sum(probs * np.log2(probs))

def ganancia_informacion(X, y, atributo):
    entropia_padre = entropia(y)
    n_total = len(y)
 
    entropia_ponderada = 0
    for valor in X[atributo].unique():
        mascara = X[atributo] == valor
        subset_y = y[mascara]
        peso = len(subset_y) / n_total
        entropia_ponderada += peso * entropia(subset_y)
 
    return entropia_padre - entropia_ponderada

#Selecciona el atributo dependiendo de que selector se le especifique 
def mejor_atributo(X, y, atributos, criterio="Ganancia"):
    funciones = {
        "Ganancia": ganancia_informacion,
        "GainRatio": gain_ratio,
        "ImpurityReduction": impurityReduction
    }

    if criterio not in funciones:
        raise ValueError(
            f"Criterio '{criterio}' no reconocido. "
            f"Usar: {list(funciones.keys())}"
        )

    mejor_atributo = None
    mejor_score = float("-inf")

    for atributo in atributos:
        score = funciones[criterio](X, y, atributo)

        if score > mejor_score:
            mejor_score = score
            mejor_atributo = atributo

    return mejor_atributo, mejor_score


# 2 Algoritmo ID3


class ID3Classifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        criterio="Ganancia",
        min_info_gain=0.0,
    ):
        self.criterio = criterio
        self.min_info_gain = min_info_gain


    def _valor_mas_comun(self, y):
        if len(y) == 0:
            raise ValueError(
                "No se puede calcular el valor más común "
                "de un conjunto vacío."
            )
        conteos = y.value_counts()
        cantidad_maxima = conteos.max()
        valores_empatados = conteos[
            conteos == cantidad_maxima
        ].index.tolist()
        if "E" in valores_empatados:
            return "E"

        return sorted(valores_empatados, key=str)[0]
    def _crear_hoja(self, valor):
        return {
            "tipo": "hoja",
            "valor": valor,
        }

    def _crear_nodo(
        self,
        atributo,
        valor_mas_comun,
    ):
        return {
            "tipo": "nodo",
            "atributo": atributo,
            "valor_mas_comun": valor_mas_comun,
            "ramas": {},
        }

    def _construir_arbol(self, X, y, atributos):
        valor_mas_comun = self._valor_mas_comun(y)

        # Caso 1: todos los ejemplos tienen el mismo resultado
        if y.nunique() == 1:
            return self._crear_hoja(y.iloc[0])

        # Caso 2: no quedan atributos para seguir dividiendo
        if len(atributos) == 0:
            return self._crear_hoja(valor_mas_comun)

        # Seleccionamos el atributo con mayor ganancia
        atributo_elegido, ganancia = mejor_atributo(
            X,
            y,
            atributos,
            criterio=self.criterio
        )

        # Caso 3: la mejor división no supera el mínimo exigido
        if atributo_elegido is None or ganancia <= self.min_info_gain:
            return self._crear_hoja(valor_mas_comun)

        # Creamos el nodo correspondiente al atributo elegido
        nodo = self._crear_nodo(
            atributo_elegido,
            valor_mas_comun
        )

        # Sacamos el atributo actual de la lista de atributos disponibles para las ramas
        atributos_restantes = [
            atributo
            for atributo in atributos
            if atributo != atributo_elegido
        ]

        # Construimos una rama para cada valor observado
        for valor in X[atributo_elegido].unique():
            mascara = X[atributo_elegido] == valor

            X_rama = X.loc[mascara, atributos_restantes]
            y_rama = y.loc[mascara]

            nodo["ramas"][valor] = self._construir_arbol(
                X_rama,
                y_rama,
                atributos_restantes
            )

        return nodo

    def fit(self, X, y):
        # ID3 necesita conocer los nombres de los atributos
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un DataFrame de pandas.")

        # Copiamos y reiniciamos los índices para mantener X e y alineados
        X = X.copy().reset_index(drop=True)
        y = pd.Series(y).copy().reset_index(drop=True)

        # Validaciones básicas
        if len(X) == 0:
            raise ValueError("El conjunto de entrenamiento está vacío.")

        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma cantidad de ejemplos.")

        if X.isna().any().any() or y.isna().any():
            raise ValueError("ID3 no admite valores faltantes en esta implementación.")

        # Información aprendida o registrada durante el entrenamiento
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array(
            sorted(y.unique(), key=str),
            dtype=object
        )

        # Construcción del árbol
        self.arbol_ = self._construir_arbol(
            X,
            y,
            list(X.columns)
        )

        return self

    def _predecir_ejemplo(self, fila):
        nodo_actual = self.arbol_
    
        # Recorremos el árbol hasta llegar a una hoja
        while nodo_actual["tipo"] != "hoja":
            atributo = nodo_actual["atributo"]
            valor = fila[atributo]
    
            # Si el valor nunca apareció durante el entrenamiento,
            # usamos el valor más común guardado en el nodo
            if valor not in nodo_actual["ramas"]:
                return nodo_actual["valor_mas_comun"]
    
            nodo_actual = nodo_actual["ramas"][valor]
    
        return nodo_actual["valor"]
    
    
    def predict(self, X):
        # Comprobamos que fit() haya sido ejecutado
        if not hasattr(self, "arbol_"):
            raise ValueError(
                "El modelo todavía no fue entrenado. Ejecute fit() antes de predict()."
            )
    
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un DataFrame de pandas.")
    
        # Comprobamos que estén los atributos utilizados al entrenar
        atributos_faltantes = [
            atributo
            for atributo in self.feature_names_in_
            if atributo not in X.columns
        ]
    
        if atributos_faltantes:
            raise ValueError(
                f"Faltan atributos necesarios: {atributos_faltantes}"
            )
    
        predicciones = []
    
        # Predecimos una fila o partido por vez
        for _, fila in X.iterrows():
            prediccion = self._predecir_ejemplo(fila)
            predicciones.append(prediccion)
    
        return np.array(predicciones, dtype=object)