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


