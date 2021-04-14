import sys

import PySimpleGUI as sg
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial.distance import cdist
from pyproj import Proj, transform
import scipy.spatial as sc
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from scipy.spatial.distance import pdist
import pyproj
from matplotlib import pyplot as plt
import numpy.random as rd
import math
import overpy as op
import urllib.error
import urllib.parse
import urllib.request
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import requests
import random
import string
import urllib, json
import csv
import sys
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import requests
import folium
import polyline
import random
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import * #Para esto hacer pip install PyQtWebEngine
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QTableWidgetItem, QFormLayout, QPlainTextEdit)
import time
import threading
from datetime import datetime
#https://github.com/pyqt/examples/tree/_/src/02%20PyQt%20Widgets
#https://stackoverflow.com/questions/52010524/widgets-placement-in-tabs
#https://realpython.com/python-pyqt-gui-calculator/


localidad = ""
numDias = 0
capacidadContenedor = 0  

headers = ['Camion','Capacidad','Velocidad','Funcionando']
data = pd.read_csv('Data/Camiones.csv', delimiter=',', header=0, names=headers)
datos = data.values.tolist() 

#print(datos)

headersContenedores = ["ID Contenedor", "Estado Inicial", "Aumento Diario"]
dataContenedores = pd.read_csv('Data/Contenedores.csv', delimiter=',', header=0, names=headersContenedores)
datosContenedores = dataContenedores.values.tolist() 

#print(datosContenedores)

datosPlanificar ={}
datosPlanificar['Localidad'] = ""
datosPlanificar['numDias'] = 0 
datosPlanificar['capacidadContenedor'] = 0 
datosPlanificar['numCamiones'] = 0
datosPlanificar['capacidadCamiones'] = []
datosPlanificar['velocidadCamiones'] = []
datosPlanificar['estadoInicial'] = []
datosPlanificar['aumentoDiario'] = []




# Elimina los warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None

"""Ponemos la URL donde están los datos a utilizar, que han tenido que ser subidos previamente a GitHub en formato CSV."""

metodoBusqueda = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
#Otro algotimo que parece ser más rápido
#metodoBusqueda = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
header = 0
nom_col = ["id","UDALERRIA_KODEA/COD_MUNICIPIO","UDALERRIA/MUNICIPIO","EKITALDIA/EJERCICIO","EDUKIONTZI_KODEA/COD_CONTENEDOR","EDUKIONTZIAREN MODELOA_CAS/MODELO CONTENEDOR_CAS","HONDAKINAREN FRAKZIOA_CAS/FRACCION DEL RESIDUO_CAS","longitude","latitude", "horaMin", "horaMax"]
sep = ','


def leerDatos(localidad):
  datos = pd.read_csv('https://raw.githubusercontent.com/Jondiii/contenedores/master/ListadoCompleto.csv', delimiter=sep, header=header, names=nom_col)

  filtro =  datos['UDALERRIA/MUNICIPIO'] == localidad
  datos = datos[filtro]
  datos.drop(columns = ["UDALERRIA_KODEA/COD_MUNICIPIO","UDALERRIA/MUNICIPIO","EKITALDIA/EJERCICIO","EDUKIONTZI_KODEA/COD_CONTENEDOR","EDUKIONTZIAREN MODELOA_CAS/MODELO CONTENEDOR_CAS","HONDAKINAREN FRAKZIOA_CAS/FRACCION DEL RESIDUO_CAS"],
             inplace = True)
  datos['longitude'] = pd.Series(datos['longitude']).str.replace(',', '.', regex=False)
  datos['latitude'] = pd.Series(datos['latitude']).str.replace(',', '.', regex=False)

  datos['longitude'] = datos['longitude'].astype(float)
  datos['latitude'] = datos['latitude'].astype(float)

  datos.index = dict(enumerate(range(0, len(datos)), 1)) # Hace que el index del dataframe vaya de 1 a len(datos). Explicación arriba.

  mediaLong = 0
  mediaLat = 0
  i = 1
  datos['horario'] = [None] * len(datos) # Crea un array vacío.
  while i <= len(datos):
    datos['horario'][i] = ((datos['horaMin'][i], datos['horaMax'][i]))
    mediaLong = mediaLong + datos['longitude'][i]#Igual es un poco carnicería geográfica, pero es lo que se nos ha ocurrido de momento
    mediaLat = mediaLat + datos['latitude'][i]
    i = 1 + i
  
  depot = {"id": 0, "longitude": mediaLong/len(datos), "latitude": mediaLat/len(datos), "horaMin" : 0, "horaMax" : 23, "horario" : [(0, 23)]}
  depot = pd.DataFrame(depot)
  datos = pd.concat([depot, datos], ignore_index = True)
  datos['recogido'] = False

  return datos

"""Se crea el modelo de datos que se va a utilizar para los cálculos. Es aquí donde se modifican los datos para poder hacer pequeñas variaciones en el ejercicio.

Hemos incluído en data todos los datos, matrices y demás necesarios para realizar los cálculos, para poder tener todo lo necesario en una única variable.
"""

def create_data_model2(localidad, capacidadCamiones, ncamiones, depot, capacidadContenedor):

    data = {}
    data['datos'] = leerDatos(localidad)
    data['distance_matrix'] = leerMatrizDistancia(localidad)
    data['time_matrix'] = crearMatrizTiempos_Enrique(data)
    data['demands'] = []
    data['vehicle_capacities'] = []
    data['num_vehicles'] = ncamiones
    data['time_windows'] = []
    data['indexes'] = []

    # crear demanda aleatoria para X num de contenedores
    i = 0
    rd.seed(2)
    while i < len(data['datos']):
      n = rd.randint(capacidadContenedor*0.1, capacidadContenedor)
      data['demands'].append(n) # Añade la carga de cada contenedor, entre 0 y 10.
      # Si vamos a trabajar con % de llenado no tiene sentido crear demandas aleatorias.
      data['time_windows'].append(data['datos']['horario'][i])
      data['indexes'].append(data['datos']['id'][i])
      i += 1

      data['num_vehicles'] = ncamiones

    #i = 0
    #while i < data['num_vehicles']:
     # data['vehicle_capacities'].append(capacidadCamiones) # Añade la capacidad de cada vehículo
      #i+=1
    data['vehicle_capacities'] = capacidadCamiones
    data['opcionales'] = pd.DataFrame(data['indexes'])

    #Por defecto 0
    data['depot'] = depot

    return data

"""#### Métodos para generar matriz de distancias
"""
def matrizDistanciasAPI(datos):
  #limit = 0
  coordenadas = ""

  longitud = round(datos['longitude'], 2).to_numpy()
  latitud = round(datos['latitude'], 2).to_numpy()

  #Transformamos de UTM a Coordenadas Geográficas (Grados)
  scrProj = pyproj.Proj(proj="utm", zone = 30, ellps="WGS84", units = "m")
  dstProj = pyproj.Proj(proj = "longlat", ellps="WGS84", datum = "WGS84")

  i = 0
  for n in datos['latitude']:
    longitud[i],latitud[i]=pyproj.transform(scrProj,dstProj, longitud[i],latitud[i])
    i +=1

  for (lat, longi) in zip(latitud, longitud):
    coordenadas = coordenadas + str(round(longi,6)) + "," + str(round(lat,6)) + ";"
  
  #while limit < 100:
  #      coordenadas = coordenadas + str(round(longitud[limit],6)) + "," + str(round(latitud[limit],6)) + ";"
  #      limit = limit + 1

  coordenadas = coordenadas[:-1]

  url = 'http://router.project-osrm.org/table/v1/driving/'+coordenadas+'?annotations=duration,distance'
  
  response = urllib.request.urlopen(url)
  result = json.load(response)
  
  '''
  req = urllib.Request(url) 
  #req.add_header('User-Agent', User_Agent)
  response = urllib.urlopen(req) 
 
  req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
  response = urllib.request.urlopen(req)
  print(req)
  '''

  #matriz = result['rows'][0]['elements'][0]["distance"]["value"] #Con esto sacamos la distancia en metros.
  matriz = pd.DataFrame(result['distances'])

  return matriz

def matrizDistanciasAPIGrande(datos):
  t = len(datos)
  if  t >= 200:
    raise Exception("La localidad no puede tener 200 o más contenedores.")
    return 0

  cont = 0
  coordenadas = ""
  coordenadas2 = ""
  arrayCoordenadasFin = []#Aquí se guardarán los strings que se usarán para sacar las distancias de los últimos contenedores

  longitud = round(datos['longitude'], 2).to_numpy()
  latitud = round(datos['latitude'], 2).to_numpy()

  #Transformamos de UTM a Coordenadas Geográficas (Grados)
  scrProj = pyproj.Proj(proj="utm", zone = 30, ellps="WGS84", units = "m")
  dstProj = pyproj.Proj(proj = "longlat", ellps="WGS84", datum = "WGS84")

  i = 0
  for n in datos['latitude']:
    longitud[i],latitud[i]=pyproj.transform(scrProj,dstProj, longitud[i],latitud[i])
    i +=1

  while cont <= 99:
    if cont == 99:
      cont2 = 100
      while cont2 < len(datos):#Guardamos las coordenadas de los 99 primeros con cada uno de los que queden
        arrayCoordenadasFin.append(coordenadas + str(round(longitud[cont2],6)) + "," + str(round(latitud[cont2],6)))
        coordenadas2 = coordenadas2 + str(round(longitud[cont2],6)) + "," + str(round(latitud[cont2],6)) + ";"#De paso, guardamos las coordenadas de solo los últimos contenedores
        cont2 = cont2 + 1
    
    coordenadas = coordenadas + str(round(longitud[cont],6)) + "," + str(round(latitud[cont],6)) + ";"
    cont = cont + 1

  coordenadas = coordenadas[:-1]
  coordenadas2 = coordenadas2[:-1]

  url = 'http://router.project-osrm.org/table/v1/driving/'+coordenadas+'?annotations=duration,distance'
  response = urllib.request.urlopen(url)
  result = json.load(response)

  matriz100 = pd.DataFrame(result['distances'])# Matriz de los 100 primeros contenedores

  url = 'http://router.project-osrm.org/table/v1/driving/'+coordenadas2+'?annotations=duration,distance'
  response = urllib.request.urlopen(url)
  result = json.load(response)

  matrizFin = pd.DataFrame(result['distances'])# Matriz de los últimos contenedores

  matrizIntermedia = pd.DataFrame(np.zeros((1, 99)))
  matrizIntermedia = matrizIntermedia[:][1]

  for string in arrayCoordenadasFin:
    url = 'http://router.project-osrm.org/table/v1/driving/'+string+'?annotations=duration,distance'
    response = urllib.request.urlopen(url)
    result = json.load(response)

    matrizTemp = pd.DataFrame(result['distances'])
    matrizIntermedia = pd.concat([matrizIntermedia, matrizTemp[:][-1:]], ignore_index=True)

  matrizIntermedia = matrizIntermedia[:][1:]#Cogemos todas las filas menos la primera (que tiene todo 0)
  matrizIntermedia = matrizIntermedia.reset_index(drop = True)

  contenedor100 = str(round(longitud[100],6)) + "," + str(round(latitud[100],6)) + ";" + coordenadas2
  url = 'http://router.project-osrm.org/table/v1/driving/'+contenedor100+'?annotations=duration,distance'
  response = urllib.request.urlopen(url)
  result = json.load(response)

  distanciasContenedor100 = pd.DataFrame(result['distances'])[:][0]#Contenedor 100 con el resto de los que sobran, solo teníamos del 0 al 99.

  #En este punto ya estarían todas las "piezas" que hacen falta, solo habría que juntarlas

  resultados = {}
  resultados['matriz100'] = matriz100
  resultados['matrizFin'] = matrizFin
  resultados['matrizIntermedia'] = matrizIntermedia
  resultados['distanciasContenedor100'] = distanciasContenedor100[1:]

  matrizTotal = pd.DataFrame(np.zeros((t, t)))
  matrizTotal.iloc[0:100, 0:100] = matriz100

  a = matrizFin
  a.index = list(range(100, t))
  a.columns = list(range(100, t))
  matrizTotal.iloc[100:t, 100:t] = a

  b = matrizIntermedia
  b.index = list(range(100, t))
  matrizTotal.iloc[100:t, 0:99] = b
  matrizTotal.iloc[0:99, 100:t] = b.transpose()

  c = distanciasContenedor100
  c.index = list(range(99, t))
  matrizTotal.iloc[99, 99:t] = c
  matrizTotal.iloc[99:t, 99] = c.transpose()
  
  resultados['matrizTotal'] = matrizTotal

  np.count_nonzero(matrizTotal==0)#Debería ser 149...

  return matrizTotal
"""La matriz de coordenadas es la siguente (las distancias se miden en m)

Método que accede en GitHub a la matriz distancia de la localidad introducida
"""

def leerMatrizDistancia(localidad):
  matriz = pd.read_csv("https://raw.githubusercontent.com/Jondiii/contenedores/master/matricesDistancia/"+localidad+".csv", header=None)

  return matriz

"""####Métodos matriz de tiempos

A partir de la matriz distancia creada en crearMatrizCoordenadas2 conseguimos una nueva matriz que refleja el tiempo en minutos que se tarda de un contenedor a otro.
"""

def crearMatrizTiempos_Enrique(data):
    distMatrix = data['distance_matrix']
    datos = data['datos']
    timeMatrix=np.zeros((distMatrix.shape))
    velocidad=20 #(km/h)
    
    for i in range(len(distMatrix)):
     for n in range(len(distMatrix[i])):       
        #Cuando consigamos meter el tiempo de servicio, quitamos el 5
        timeMatrix[i][n] = 5 + getMin(distMatrix[i][n] / (velocidad/3.6))

    return timeMatrix

"""#### Métodos intermedios

El siguiente método creará el Routing Model. Para ello, primero se tiene que crear el Index Manager, los cuales se utilizan para señalizar los nodos por los que se estén pasando. Primero se pasa el n mero de contenedores, luego los vehículos y finalmente el punto de partida.

Una vez tenemos el manager, creamos el modelo, que es quien se encarga de todos los cálculos, para lo que basta con pasarle el manager creado anteriormente.

También sería posible configurar el orden de entrega de los paquetes (mismo orden que el de recogida o el inverso). Aunque no nos interese para este problema, podría llegar a ser útil en el futuro.
"""

def creaRoutingModel(data):
  #Primero se crea el Routing Index Manager
  manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
  #Después se crea el Routing Model
  routing = pywrapcp.RoutingModel(manager)

  return routing, manager

"""Este método se usa para crear dimensiones. Las dimensiones son objetos que el solver utiliza para registrar las distintas cantidades que tiene un vehículo (capacidad, tiempo etc). Se puede encontrar más información sobre estas [aquí](https://developers.google.com/optimization/routing/dimensions#slack_variables)."""

def creaDimensiones(routing, manager, data):
# Esta función callback toma dos localizaciones y devuelve la distancia entre ellas.
  def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

  # Crea un índice interno al método callback que devuelve la distancia.
  transit_callback_index = routing.RegisterTransitCallback(distance_callback)

  # Coste de ir de un punto a otro (en este caso solo la distancia).
  # Existe una variante que permite establecer distintas velocidades a cada vehículo.
  # Se calcula utilizando la callback creada anteriormente. Asumimos que podemos modificar
  # la callback a nuestro antojo para obtener resultados distintos.
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

  # Obtiene la carga que tiene un punto concreto.
  def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return data['demands'][from_node]

  # No tenemos una time_callback, pero no sé si sería necesaria para sacar rutas.
  """def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data["time_matrix"][from_node, to_node]"""

  # Crea un índice interno al método callback que devuelve la carga.
  demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

  # Añade las dimensiones.
  routing.AddDimensionWithVehicleCapacity(
      demand_callback_index, 
      0, # Slack. Indica cuánto tiempo está un vehículo en un punto concreto.
      data['vehicle_capacities'],  # Capacidades de los vehículos
      True,  # True para que la carga inicial de cada vehículo sea 0.
      'Capacity')

"""Método igual que el anterior, pero que se usará en caso de quer añadir la restricción de las Time Windows. No produce resultados correctos (de momento)."""

def creaDimensionesTW(routing, manager, data):
# Esta función callback toma dos localizaciones y devuelve la distancia entre ellas.
  def distance_callback(from_index, to_index):
      from_node = manager.IndexToNode(from_index)
      to_node = manager.IndexToNode(to_index)
      return data['distance_matrix'][from_node][to_node]

  # Crea un índice interno al método callback que devuelve la distancia.
  transit_callback_index = routing.RegisterTransitCallback(distance_callback)

  # Coste de ir de un punto a otro (en este caso solo la distancia).
  # Existe una variante que permite establecer distintas velocidades a cada vehículo.
  # Se calcula utilizando la callback creada anteriormente. Asumimos que podemos modificar
  # la callback a nuestro antojo para obtener resultados distintos.
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

  # Obtiene la carga que tiene un punto concreto.
  def demand_callback(from_index):
      from_node = manager.IndexToNode(from_index)
      return data['demands'][from_node]

  # Crea un índice interno al método callback que devuelve la carga.
  demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

  # Añade las dimensiones.
  routing.AddDimensionWithVehicleCapacity(
      demand_callback_index, 
      0, # Slack. Indica cuánto tiempo está un vehículo en un punto concreto.
      data['vehicle_capacities'],  # Capacidades de los vehículos
      True,  # True para que la carga inicial de cada vehículo sea 0.
      'Capacity')
  
  # Callback que recupera el tiempo de viaje entre dos nodos
  def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return getMin(data['time_matrix'][from_node][to_node])
    
  # Mirar por qué se usa SetArcCostEvaluatorOfAllVehicles dos veces (aquí por segunda vez).
  transit_callback_index = routing.RegisterTransitCallback(time_callback)
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

  time = 'Time'
  routing.AddDimension(
    transit_callback_index,
    30,  # allow waiting time
    30,  # maximum time per vehicle
    False,  # Don't force start cumul to zero.
    time)
  time_dimension = routing.GetDimensionOrDie(time)
  
  # Enumerate añade a cada elemento de una lista un id (que en este caso se guardará en location_idx)
  for location_idx, time_window in enumerate(data['time_windows']):
    if location_idx == 0:
        continue
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1])) #Intentar poner el int() ese más "elegante".

  for i in range(data['num_vehicles']):
    routing.AddVariableMinimizedByFinalizer(
        time_dimension.CumulVar(routing.Start(i)))
    routing.AddVariableMinimizedByFinalizer(
        time_dimension.CumulVar(routing.End(i)))

"""Se crean parámetros por defecto y se elige el método de búsqueda. Esto es lo más interesante de este trozo de código, pues puede influenciar en gran medida el resultado. La lista de métodos se puede encontrar [aquí](https://developers.google.com/optimization/routing/routing_options#first_sol_options). Algunos notables:

"""

def creaParametrosBusqueda():
  search_parameters = pywrapcp.DefaultRoutingSearchParameters()
  
  #Por culpa del limite de tiempo a veces saltaba un error, de momento lo comentamos.
  #Limit in seconds to the time spent in the search.
  search_parameters.time_limit.seconds = 10

  #Limit to the number of solutions generated during the search.
  #search_parameters.solution_limit = 100

  search_parameters.first_solution_strategy = (metodoBusqueda)

  return search_parameters

"""####Utilidades

Función para convertir el tiempo a el formaro hh:mm:ss.
"""

def getTime(t):
  if t == 0:
    return '00:00:00'
    
  else:
    t = round(t, 2)

    horas = str(t // 60)[0].zfill(2)
  
    min = str(math.floor(t % 60)).zfill(2)

    seg = t % 60 - int(t % 60)
    seg = str(math.floor(((seg/100)*60)*100)).zfill(2)
    return horas + ':' + min + ':' + seg

"""Función que convierte nuestros números int a floats para que tengan sentido a la hora de imprimirlos. Actualmente, OR-Tools no puede utilizarse con floats ([fuente](https://github.com/google/or-tools/issues/2149)) por lo que tenemos que usar ints y luego ponerles la coma. Esto es algo que debería solucionarse en la versión 8.0 de OR-Tools, así que de momento usaremos este método para salir del paso. FLTV8"""

def toFloat(n):
  return n/100

"""Método para pasar de segundos a minutos. FLTV8"""

def getMin(seg):
  min = (round(seg)//60)
  if round(seg)%60 >0.5:
    min += 1
  
  return int(min)

"""Método que recibe los datos y devuelve las latitudes, longitudes y el depot"""

def getCoordenadas(data):
  longitud = (data['datos']['longitude'].to_numpy(dtype=float)).copy()
  latitud = (data['datos']['latitude'].to_numpy(dtype=float)).copy()
  #Transformamos de UTM a Coordenadas Geográficas (Grados)
  scrProj = pyproj.Proj(proj="utm", zone = 30, ellps="WGS84", units = "m")
  dstProj = pyproj.Proj(proj = "longlat", ellps="WGS84", datum = "WGS84")

  i = 0
  for n in data['datos']['latitude']:
      longitud[i],latitud[i]=pyproj.transform(scrProj,dstProj, longitud[i],latitud[i])
      i +=1

  return latitud, longitud, [latitud[0], longitud[0]]


"""####Varios

Método que toma un DataFrame y sustituye sus índices por un rango de valores que va de `value` a `n+value` (siendo n la longitud del DataFrame - 1). Devuelve el DataFrame con los índices cambiados y devuelve los índices antiguos.
"""

def reseteaIndices(data, value):  
  oldIndex = list(data.index)
  newIndex = pd.Series(np.arange(0+value,len(data)+value))
  data.index = newIndex

  return data, oldIndex

def reseteaIndices(data, value, new = False):  
  oldIndex = list(data.index)
  newIndex = pd.Series(np.arange(0+value,len(data)+value))
  data.index = newIndex

  if new == True:
    return data, newIndex
  else:
    return data, oldIndex

"""Método que recibe el DataFrame datos y devuelve el mismo DaraFrame pero solo con los contenedores que no hayan sido recogidos aún."""

def filtro(data):
  return data[data['recogido'] == False]

"""Método que recibe un dataframe de una única fila y devuelve una lista. Existe df.values.tolist() pero esto devuelve una lista donde cada valor es un array de longitud 1, y eso da problemas al intentar usar los métodos de OR Tools."""

def dfToList(dataFrame, ceros = False):
  primeraVez = True
  if dataFrame.shape[1] != 1:
    print("El dataFrame no tiene una única fila.")

  else:
    lista = []
    for n in dataFrame.values.tolist():
      if (n[0] != np.nan) & (not ceros):
        lista.append(n[0])
      else:
        if (ceros & ((n[0] != np.nan) & ((n[0] != 0) or primeraVez))):
          lista.append(n[0])
          primeraVez = False

    return lista

"""Método que recibe la capacidad máxima de los contenedores y un dataframe con el porcentaje de llenado de cada contenedor. De ahí calcula las demandas (y las devuelve). """

def calculaDemandas(capacidadCont, df, ceros=False):
  df = dfToList(df, ceros)
  demands = []
  for n in df:
    demands.append(capacidadCont*(n/100))

  return demands

def procesaVector(vector, separadorV):
  str1 = ','.join(str(e) for e in vector)
  return str1.split(separadorV)

def fromCharToInt(vector):
  return [int(s) for s in vector]

"""####Crear nuevo DataModel

Crea un nuevo DataModel eliminando de `data` todos los datos relacionados con los índices de `contARecoger`.
"""

def newDatamodel(data, contARecoger):

  indicesDrop = []
  cont = 0
  for i in dfToList(contARecoger):
    if i == 0.0:
      indicesDrop.append(cont)
    cont += 1

  newData = {}
  newData['datos'], newIndexes = reseteaIndices(data['datos'].drop(indicesDrop), 0, True)

  #Dropeamos filas y columnas
  newData['distance_matrix'] = (pd.DataFrame(data['distance_matrix'])).drop(indicesDrop)
  newData['time_matrix'] = (pd.DataFrame(data['time_matrix'])).drop(indicesDrop)
  newData['distance_matrix'] = newData['distance_matrix'].drop(indicesDrop, axis = 1) #1 = columnas
  newData['time_matrix'] = newData['time_matrix'].drop(indicesDrop, axis = 1)

  # De esta forma estamos modificando el original 
  
  copiaDemandas = data['demands'].copy()
  
  for n in indicesDrop:
    copiaDemandas.remove(data['demands'][n])
    
  newData['demands'] = copiaDemandas
  newData['vehicle_capacities'] = data['vehicle_capacities'].copy()
  newData['num_vehicles'] = data['num_vehicles']
  newData['indexes'] = list(newData['datos'].index)

  # Teniendo los índices cambiamos los de ambas matrices
  newData['distance_matrix'].index = newData['indexes']
  newData['time_matrix'].index = newData['indexes']
  newData['distance_matrix'].columns =  newData['indexes']
  newData['time_matrix'].columns = newData['indexes']

  # Pasamos las matrices a listas para que las lea bien. Igual el paso anterior es innecesario.
  newData['distance_matrix'] = newData['distance_matrix'].values.tolist()
  newData['time_matrix'] = newData['time_matrix'].values.tolist()

  newData['depot'] = 0

  return newData

"""####Sacar plan

Método que calcula un plan inicial aleatorio recibiendo el número máximo de días y el número de contenedores.
"""

def randomPlan(nCont, nDias):

  rng = np.random.default_rng(1)
  plan = pd.DataFrame(rng.integers(1, nDias+1, size=nCont))

  return plan

"""Método que recibe data y saca un estado inicial, un aumento diario (ambos semi-aleatorios) y un plan. Habría que reajustarlo en caso de que cada camión tenga distintas capacidades o que se quiera usar un número distinto de camiones cada día."""
## CREO QUE NO SE USA 
def sacarPlan(data, sizeCont, nDias, capacidadTotal, estadoI, aumentoD):

  #estadoI = [0]
  #aumentoD = [0]
 
  estadoI = estadoI.drop(0)
  aumentoD = aumentoD.drop(0)
  print("total truck capacity: ", capacidadTotal)
  plan = [0]
  '''
  i = 0
  rd.seed(1)
  while i < len(data['datos'])-1:
    eI = rd.choice((30, 40, 50, 60))
    aD = rd.choice((10, 20, 30))
    estadoI.append(eI) # Añade la carga de cada contenedor (30, 40, 50, 60)
    aumentoD.append(aD)
    # Si vamos a trabajar con % de llenado no tiene sentido crear demandas aleatorias.
    plan.append(nDias+1)
    i += 1
  '''
  i = 1
  estadoDF = pd.DataFrame(estadoI)
  data['demands'] = dfToList(estadoDF)
  aumentoDF = pd.DataFrame(aumentoD)

  while i <= nDias:
    recogido = 0
    cont = 0
    newEstado = estadoDF

    for n in newEstado[0]:
      if ((n + aumentoDF[0][cont]) > 100):#Contenedor que desbordará mañana
        if (recogido + n) <= capacidadTotal:
          recogido = recogido + n
          newEstado[0][cont] = 0 #Se ha recogido el contenedor.
          plan[cont] = i

        else:
          #TODO ¿Qué hacer si el contenedor va a desbordar pero no puede ser recogido este dia?
          pass

      elif (((recogido + n) <= capacidadTotal) & (n > 40) & (plan[cont]==nDias+1)):#Dejamos los que tengan menos de 20% para "rellenar" posibles huecos
        recogido = recogido + n
        newEstado[0][cont] = 0 #Se ha recogido el contenedor.
        plan[cont] = i
       
      cont += 1
    
    cont = 0
    for n in newEstado[0]:
      if (((recogido + n) <= capacidadTotal) & (n <= 400) & (n != 0) & (plan[cont]==nDias+1)):
        recogido = recogido + n
        newEstado[0][cont] = 0
        plan[cont] = i
      cont =+ 1
    

    estadoDF = newEstado.copy() + aumentoDF
    i += 1  

  plan = pd.DataFrame(plan)
  #print("plan: ", dfToList(plan))

  return plan

"""###Visualizar resultados

####Prints

Saca las rutas, el tiempo de llegada a cada contenedor, la carga en cada punto de la ruta, la carga total y la distancia total.

Además, se ha juntado con el método que guarda las rutas.
"""

def print_solution_detail(data, manager, routing, solution):
    print("\nSOLUCIÓN\n")
    total_distance = 0
    total_load = 0
    total_time = 0
    listaRutas = []

    for vehicle_id in range(data['num_vehicles']):
      index = routing.Start(vehicle_id)
      plan_output = 'Route for vehicle {} - (start time 00:00:00):\n'.format(vehicle_id)
      route_distance = 0
      route_load = 0
      route_time = 0
      ruta = []

      while not routing.IsEnd(index):
          node_index = manager.IndexToNode(index)
          route_load += data['demands'][node_index]
          plan_output += ' {0} min  Recogida  contenedor #{1}  ({2} tonelada(s) - total: {3}) \n'.format(getTime(route_time), node_index, toFloat(data['demands'][node_index]), toFloat(route_load))
          ruta.append(node_index)
          #Para la distancia
          previous_index = index
          index = solution.Value(routing.NextVar(index))
          route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
          #Para el timempo
          previous_node_index = node_index
          node_index = manager.IndexToNode(index)
          route_time += data['time_matrix'][previous_node_index][node_index];

      ruta.append(data['depot'])  
      listaRutas.append(ruta)
      plan_output += ' {0} Llegada Depot #{1}\n\n'.format(getTime(route_time), manager.IndexToNode(index))
      plan_output += 'Distance of the route: {}m\n'.format(route_distance)
      plan_output += 'Load of the route: {}t\n'.format(toFloat(route_load))
      plan_output += 'Total time: {}\n\n'.format(getTime(route_time))
      print(plan_output)
      total_distance += route_distance
      total_load += route_load
      total_time += route_time
    
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(toFloat(total_load)))

    return listaRutas

"""Método que conseguirá todas las rutas pero no las imprimirá."""

def getRutas(data, manager, routing, solution):
    total_distance = 0
    total_load = 0
    total_time = 0
    listaRutas = []
    listaTiempos = []
    listaCargas = []
    listaLlenadoC = []
    resultado = {}

    for vehicle_id in range(data['num_vehicles']):
      index = routing.Start(vehicle_id)
      route_distance = 0
      route_load = 0
      route_time = 0
      ruta = []
      tiempos = []
      cargas = []
      llenadoC = []

      while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route_load += data['demands'][node_index]
        
        #Guardamos datos de las rutas
        ruta.append(node_index)
        tiempos.append(route_time)
        cargas.append(route_load)
        llenadoC.append(data['demands'][node_index])

        #Para la distancia
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        #Para el timempo
        previous_node_index = node_index
        node_index = manager.IndexToNode(index)
        route_time += data['time_matrix'][previous_node_index][node_index]

      ruta.append(data['depot'])
      tiempos.append(route_time)
      cargas.append(route_load)
      llenadoC.append(0)

      listaRutas.append(ruta)
      listaTiempos.append(tiempos)
      listaCargas.append(cargas)
      listaLlenadoC.append(llenadoC)

      total_distance += route_distance
      total_load += route_load
      total_time += route_time
    
    resultado['listaRutas'] = listaRutas
    resultado['listaTiempos'] = listaTiempos
    resultado['listaCargas'] = listaCargas
    resultado['listaLlenadoC'] = listaLlenadoC
    resultado['total_distance'] = total_distance
    resultado['total_load'] = total_load
    resultado['total_time'] = total_time

    return resultado

"""Este método hace print de los KPIs establecidos (WIP)."""

def sacaKPIs(data, limite):
  print("Contenedores casi al límite de su capacidad ({}):".format(limite))

  cont = 0
  i = 0
  while i < len(data['demands']):
    llenado = data['demands'][i]/400 
    if (llenado >= 0.8):
      cont += 1
      print("Contenedor #{0}: {1}".format(data['indexes'][i], round(llenado, 2)))
    i += 1

  if cont == 0:
    print("Ningún contenedor está al límite de su capacidad.")
  else:
    print("\nTotal: {0} ({1}%)".format(cont, round((cont*100)/len(data['demands'])), 2))

"""Método que imprime todo lo que haya en un diccionario."""

def imprimeData(data):

  for key in data:
    print('{}: '.format(key))
    if not isinstance(data[key], int):
      print('Tamaño: {}'.format(len(data[key])))
    print(data[key])
    print('\n')

"""####Visualización"""

def representarContenedores(listaRutas, data, localidad, dia, resultado, demanda):

    #Leer coordenadas.
    datos = leerDatos(localidad)

    longitud = datos['longitude'].to_numpy()
    latitud = datos['latitude'].to_numpy()


    #Transformamos de UTM a Coordenadas Geográficas (Grados)
    scrProj = pyproj.Proj(proj="utm", zone = 30, ellps="WGS84", units = "m")
    dstProj = pyproj.Proj(proj = "longlat", ellps="WGS84", datum = "WGS84")

    i = 0
    for n in datos['latitude']:
        longitud[i],latitud[i]=pyproj.transform(scrProj,dstProj, longitud[i],latitud[i])
        i +=1

    d = dia + 1
    title = ("Rutas del dia %i" % (d)) 
    plt.figure(figsize=(10, 10))
    plt.suptitle(title,  ha='center')
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
  
    demandas = demanda[dia]
    demandas = np.array(demandas)
    correctas = demandas<90
    limite = (90 <= demandas) & (demandas < 100)
    desbordadas = demandas>100

    plt.scatter(longitud[correctas],latitud[correctas], marker="o", color="blue")
    plt.scatter(longitud[limite],latitud[limite], marker="o", color="yellow")
    plt.scatter(longitud[desbordadas],latitud[desbordadas], marker="o", color="red")
  
    for n,txt in enumerate(demandas):
     plt.annotate(txt, (longitud[n], latitud[n]))
    
    listaRutasEditada = []
  
    for ruta in listaRutas:
      #Si ponemos el marker los puntos se ponen del color de las rutas sí o sí.
      plt.plot(longitud[ruta], latitud[ruta]) 
      if (len(ruta) > 2): 
        listaRutasEditada.append(ruta)
      
    plt.title(listaRutasEditada, y = 1.05,  ha='center')
    plt.show()


"""####Mapeado"""


def get_route(ruta, lat, longi):
  strRuta = ""
  for contenedor in ruta:
      strRuta = strRuta + str(longi[contenedor]) + "," + str(lat[contenedor]) + ";"

  strRuta = strRuta[:-1]

  url = 'http://router.project-osrm.org/route/v1/driving/'+strRuta+'?annotations=distance,duration'

  r = requests.get(url)
  res = r.json()

  routes = polyline.decode(res['routes'][0]['geometry'])
  depot = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
  distance = res['routes'][0]['distance']
  
  out = {'route':routes,
          'depot':depot,
          'distance':distance,
          'ruta':ruta,
          'lat':lat,
          'long':longi
        }

  return out

def get_map(lat, longi, depot, cCamiones, cContenedor, rutas, tiempos, llenados, contenedores):
  mapas = []
  colores = ('red', 'green', 'blue', 'yellow', 'deeppink', 'darkmagenta', 'orange', 'mediumspringgreen',
    'darkturquoise', "teal", "navy")
  m = folium.Map(location=depot,
              zoom_start=13)

  n = 0
  for ruta, horas, llenado, capacidad, estadoCont in zip(rutas, tiempos, llenados, cCamiones, contenedores):#Zip parará cuando uno de los dos termine.
    if len(ruta)>2:
      n += 1
      out = get_route(ruta, lat, longi)
      feature_group = folium.FeatureGroup(name="Ruta "+str(n))

      folium.PolyLine(
          out['route'],
          weight=8,
          color=colores[n-1],
          opacity=0.8
      ).add_to(feature_group)

      folium.Marker(
          location=out['depot'],
          icon=folium.Icon(icon='play', color='green')
      ).add_to(feature_group)

      locations = {
          'lat': out['lat'],
          'long': out['long']
      }

      locationList = pd.DataFrame(locations)
      locationList = locationList.values.tolist()

      i = 0
      porcentajeLlenado = 0
      for point in out['ruta']:
          porcentajeLlenado = round((llenado[i]/capacidad)*100, 2)
          folium.Marker(
              locationList[point], tooltip=str(i),
              popup="Hora planificada: {0}\nParada {1} del camión {2}\nContenedor al {3}%\nCamión al {4}%".format(getTime(horas[i]), i, n, round((estadoCont[i]/cContenedor)*100,2), porcentajeLlenado)
          ).add_to(feature_group)
          i += 1

      
      feature_group.add_to(m)
      
  folium.LayerControl().add_to(m)
      
  mapas.append(m)

  return mapas


"""### Función principal

Esta función generará las demandas iniciales, así como el aumento diario de forma semi-aleatoria.
"""

def init(nCont):
  estadoI = [0]
  aumentoD = [0]
  
  i = 0
  rd.seed(1)
  while i < nCont-1:
    eI = rd.choice((30, 40, 50, 60))
    aD = rd.choice((10, 20, 30))
    estadoI.append(eI) # Añade la carga de cada contenedor (30, 40, 50, 60)
    aumentoD.append(aD) # Añade el aumento diario de cada contenedor (10, 20, 30)
    # Si vamos a trabajar con % de llenado no tiene sentido crear demandas aleatorias.
    i += 1

  estadoDF = pd.DataFrame(estadoI)
  aumentoDF = pd.DataFrame(aumentoD)

  return estadoDF, aumentoDF

def solucionaProblema(data):
  # Crea el gestor de índices y el modelo.
  routing, manager = creaRoutingModel(data)
  
  ### Puede se pueda añadir algo!!

  # Crea dimensiones, que guardarán información de cada nodo.
  creaDimensiones(routing, manager, data)

  # Crea unos parámetros de búsqueda definidos por defecto para comenzar la búsqueda.
  search_parameters = creaParametrosBusqueda()

  # Solucionar el problema # No pasa de aquí con las TW.
  solution = routing.SolveWithParameters(search_parameters)
  #HABRÍA QUE CAMBIAR ESTO PARA QUE DEVUELVA SOLUTION Y TODO LO NECESARIO PARA RESOLVER EL PROBLEMA

  if solution:
    #listaRutas = print_solution_detail(data, manager, routing, solution)
    resultado = getRutas(data, manager, routing, solution)
    #representarContenedores(listaRutas,data,localidad)
    #sacaKPIs(data, limite)
    return solution, manager, routing, resultado

  else:
     
    #return solution, manager, routing
    return solution, manager, routing, None

  #return solution, manager, routing

"""#### Funcion"""

def funcion(data, plan, estadoContenedores, aumentoDiario, capacidadTotal,localidad,tamanyoContenedor):

  text_file = open("Data/sample.txt", "w")
  now = datetime.now()
  dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
  text_file.write("Fecha de creacion: {}\n\n".format(dt_string))
  if (len(data['datos'])!=len(plan)):
    raise Exception("El número de contenedores en el plan y en el data no coinciden")

  i = 0
  # print("max1: ",plan.max()[0])
  #print(plan.max())

  numDias = int(plan.max())
  results = []
  costes = []
  dataOriginal = data
  aumentoDiario = (aumentoDiario * tamanyoContenedor)/100
  #capacidadTotal = data['num_vehicles'] * data['vehicle_capacities'][0]
  demandas = []

  #print("Estado inicial: ",(dfToList(estadoContenedores)))
  #print("#Cantidad de toneladas que tienen los contenedores antes de empezar a recoger\n")
  #text_file.write("Estado inicial: {} \n".format(dfToList(estadoContenedores)))
  #text_file.write("#Cantidad de toneladas que tienen los contenedores antes de empezar a recoger\n")
  

  while i <= numDias:
    # tantos "Números altos" como días vayamos a planificar (+1)
    costes.append(10000)
    i += 1
  
  dia = 1
  text_file.write("- - - - - - - - - - - - - - - - - -  \n")
  text_file.write("PLANIFICACION de {} \n".format(localidad))
  text_file.write("- - - - - - - - - - - - - - - - - -  \n")

  while dia <= numDias:
    text_file.write("\nDIA: {}\n".format(dia))
    text_file.write("- - - - - - - \n")
    
    indicesRecoger = []

    indices = []
    
    plan[0][0] = dia
    contenedoresARecoger = plan[plan == dia]

    contenedoresARecoger =  (contenedoresARecoger.replace(np.nan, 0))/dia
    demanda = contenedoresARecoger*(estadoContenedores)
    demanda = (demanda.replace(np.nan, 0.0))
    #data['demands'] = demanda

    #print("Cont a recoger: ", dfToList(contenedoresARecoger))
    #text_file.write("Cont a recoger: {}\n".format(dfToList(contenedoresARecoger)))
    contador = int(0)
    for c in dfToList(contenedoresARecoger): 
      if (c == 1): 
        indices.append(contador)
      contador  += 1
    
    text_file.write("#Indice de los contenedores a recoger el dia {}\n".format(dia))
    text_file.write("Indices: {}\n".format(indices))
    #print("#Indice de los contenedores a recoger el día" , dia)
    #print("Indices:" , indices)
    
    demandaActual = []
    #print("Demanda: ", dfToList(demanda))
    for d in dfToList(demanda): 
      if (d != 0): 
        demandaActual.append(d)
    
    text_file.write("Demanda: {}\n".format(demandaActual))
    #print("Demanda: ", (demandaActual))

    cont = 0
    for i in plan:
      if  i == dia:
        indicesRecoger.append(cont)
      cont += 1

    #print("Indice: ", indicesRecoger)

    data = newDatamodel(dataOriginal, contenedoresARecoger)

    if (demanda[demanda>tamanyoContenedor].isnull().sum().sum()!=len(demanda)): 
      #Salimos del bucle, dejar de planificar el resto de días.
      desborde = list(np.where(demanda>tamanyoContenedor)[0])
      text_file.write("Contenedor(es) desbordado(s): {}\n".format(desborde))
      text_file.write("#contenedor desborda cuando demanda > 100\n")
      #print("Contenedor(es) desbordado(s):", desborde)
      #print("#contenedor desborda cuando demanda > 100")
      
      break

    else:

      solucion, manager, routing, resultado = solucionaProblema(data)
  
      total = 0
      for n in data['demands']:
        total = total + n
      #print("Total demandas: ", np.sum(demanda))
      
      if solucion:
        costes[dia-1] = -10000 + (resultado['total_distance']/1000)  
        # / 1000 para pasar a km 
      
        estadoContenedores = estadoContenedores * (1-contenedoresARecoger)
       
        estadoContenedores = estadoContenedores + aumentoDiario
        

        results.append(resultado)
        
        #print("  Solución encontrada. Resultado: ", resultado)
        nuevasRutas = []
        #print("Rutas: ")
        text_file.write("Rutas: \n")
        #Solo imprime las rutas que pasen 
        for element in resultado['listaRutas']:
          nuevasRutas.append([indices[i] for i in element])
          if (len(element) > 2):
            ##print([indices[i] for i in element])
            text_file.write(format([indices[i] for i in element]))
          #text_file.write("\n")
       
        text_file.write("\n - Distancia Total: {} m\n".format(resultado['total_distance']))
        text_file.write(" - Carga Total: : {} toneladas\n".format(resultado['total_load']))
        text_file.write(" - Distancia Total: {} min\n".format(resultado['total_time']))
        #print("\n - Distancia Total: ", resultado['total_distance'], "m")
        #print(" - Carga Total: ", resultado['total_load'], "toneladas")
        #print(" - Tiempo Total: ", resultado['total_time'], "min")
        #print(nuevasRutas)
        resultado['listaRutas'] = nuevasRutas
  
      else:  # Si el plan para ese día no es válido, no se puede recoger con los camiones que tenemos
        costes[dia-1] = int(costes[dia-1]+estadoContenedores.sum())
        estadoContenedores = estadoContenedores + aumentoDiario

        text_file.write("  No tengo suficientes camiones: {}\n".format(costes[dia-1]))
        #print("  No tengo suficientes camiones: ", costes[dia-1])
        
    dataOriginal['demands'] = dfToList(estadoContenedores)
    demandas.append(dfToList(estadoContenedores))
    # Ponemos a 0 las demandas de los contenedores que han sido recogidos.
    for i in indices:
      #estadoContenedores[i] = 0
      dataOriginal['demands'][i] = 0

    estadoContenedores = dataOriginal['demands']
    estadoContenedores = pd.DataFrame(estadoContenedores)
   # demandas.append(dfToList(estadoContenedores))
   
    dia += 1

  text_file.write("\nCoste: {}\n".format(costes))    
  text_file.close()

  return costes, results, demandas

"""####FuncionCostes"""

def funcionCostes(data, plan, estadoContenedores, aumentoDiario, capacidadTotal, tamanyoContenedor):
  if (len(data['datos'])!=len(plan)):
    raise Exception("El número de contenedores en el plan y en el data no coinciden")

  i = 0
  numDias = plan.max()[0]

  results = []
  costes = []
  dataOriginal = data
  aumentoDiario = (aumentoDiario * tamanyoContenedor)/100
  #print("Estado inicial: ", dfToList(estadoContenedores))
  
  while i <= numDias:
    # tantos "Números altos" como días vayamos a planificar (+1)
    costes.append(10000)
    i += 1
  
  dia = 1

  while dia <= numDias:
    #print("\nDIA: ", dia)
    indicesRecoger = []

    plan[0][0] = dia
    contenedoresARecoger = plan[plan == dia]

    contenedoresARecoger =  (contenedoresARecoger.replace(np.nan, 0))/dia
    demanda = contenedoresARecoger*estadoContenedores
    #print("Cont a recoger: ", dfToList(contenedoresARecoger))
    #print("Demanda: ", dfToList(demanda))
    
    cont = 0
    for i in plan:
      if  i == dia:
        indicesRecoger.append(cont)
      cont += 1
    #print("DIA {0}, cont a recoger {1}".format(dia, len(indicesRecoger)))#ESTA LENGTH DA 0
    data = newDatamodel(dataOriginal, contenedoresARecoger)

    if (demanda[demanda>tamanyoContenedor].isnull().sum().sum()!=len(demanda)): 
      #Salimos del bucle, dejar de planificar el resto de días.
      desborde = list(np.where(demanda>tamanyoContenedor)[0])
      #print("OJO, DESBORDE")
      for d in desborde:
        costes[dia-1] = costes[dia-1]+dfToList(estadoContenedores)[d]

      #print("  Contenedor(es) desbordado(s) día {0}: {1} - coste: {2}".format(dia, desborde, costes[dia-1]))

      break

    else:
      solucion, manager, routing, resultado = solucionaProblema(data)

      if solucion:
        costes[dia-1] = -10000 + (resultado['total_distance']/1000)  
        # / 1000 para pasar a km 
        #estadoContenedores = estadoContenedores * (1-contenedoresARecoger)
        estadoContenedores = estadoContenedores + aumentoDiario

        results.append(resultado)
        #print("  Solución encontrada día {0}. Coste: {1}. Resultado: {2}".format(dia, costes[dia-1], resultado))


      else:  # Si el plan para ese día no es válido, no se puede recoger con los camiones que tenemos
        estadoContenedores = estadoContenedores + aumentoDiario
        costes[dia-1] = int(costes[dia-1]+estadoContenedores.sum())
        print("  No se ha encontrado solución en el día {0}, coste: {1}".format(dia,costes[dia-1]))
        
    # Ponemos a 0 las demandas de los contenedores que han sido recogidos.
    for i in indicesRecoger:
      dataOriginal['demands'][i] = 0
   
    dataOriginal['demands'] = dfToList(estadoContenedores)

    dia += 1

  return costes, results

"""### Optimización"""

# Otro método que suma o resta 1 de forma semi aleatoria.
def suma2(plan, nCont, diaMax):
  rng = np.random.default_rng()# Poner un 1 en el paréntesis para que siempre salgan los mismos valores.
  mod = int(rng.integers(low = 1, high = nCont))
  mod2 = int(rng.integers(low = 1, high = nCont))

  plan = dfToList(plan) #Un poco estúpido este paso y el de después del if else, pero es que si no da error :/

  if (mod2 % 2) == 0: # Si mod2 es par, +1 a plan[mod]
    if (plan[mod] == diaMax): # Si ese contenedor ya va a ser recogido el último día, restar 1 en su lugar
      plan[mod] = plan[mod]-1
    else:
      plan[mod] = plan[mod]+1

  else:               # Si mod2 es impar, -1 a plan[mod]
    if (plan[mod] == 1): # Si ese contenedor ya va a ser recogido el primer día, sumar 1 en su lugar
      plan[mod] = plan[mod]+1
    else:
      plan[mod] = plan[mod]-1

  plan = pd.DataFrame(plan) #El otro paso estúpido

  # Hacemos que ningún contenedor del plan pueda ser menor que 1 ni mayor que el día máximo
  plan[0][np.where(plan<=0)[0]] = 1
  plan[0][np.where(plan>diaMax)[0]] = diaMax

  return plan

def optimizacion(planInicial, costeInicial, ncontenedores, estadoContenedores, aumentoDiario, data, iteraciones, capacidadTotal, vecindario, tamanyoContenedor): 
  plan = []
  planes = []
  listaCostes = []
  
  i = 1
  while (i <= iteraciones):

    n = 1
    while (n < vecindario):
      #if imprime:
        #print("\n·············")
        #print("Iteración: ", i)
        #print("Vecindario: {0}/{1}".format(n, ncontenedores-2)) 

      #plan = suma(planInicial, n, planInicial.max()[0])
      plan = suma2(planInicial, ncontenedores, planInicial.max()[0])
      costes, _ = funcionCostes(data, plan, estadoContenedores, aumentoDiario, capacidadTotal, tamanyoContenedor)
    
      coste = 0 
      for c in costes: 
       coste += c

      #if imprime:
        #print("Plan: ", dfToList(plan) )
        #print("Coste: ", coste)

      planes.append(plan)
      listaCostes.append(coste)
      n += 1

    mejorCoste = np.min(listaCostes)
    maxPos = listaCostes.index(min(listaCostes))
    
    if (mejorCoste < costeInicial):
      costeInicial = mejorCoste
      planInicial = planes[maxPos]
    

    print("\n\nMejor coste iteración {0}: {1}\n".format(i, mejorCoste))

    i += 1
  
  
  return planInicial, costeInicial


'''
------------------------------------
          V E N T A N A   
------------------------------------
'''
listaLocalidades = pd.read_csv("https://raw.githubusercontent.com/Jondiii/appContenedores/master/Data/localidades.txt",delimiter=sep, header=header)


class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)
        
        self._dataCamiones = datos#Esto no se llega a usar, creo
        self._dataContenedores = datosContenedores
        self._planThread = threading.Thread(target=self.planificar)
        self._planThread.daemon = True
        self.setFixedWidth(700)
        self.setFixedHeight(600)
  
        self.createBottomLeftTabWidget()
        
        topLayout = QHBoxLayout()
        topLayout.addStretch(1)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addWidget(self.bottomLeftTabWidget)
        
        lowerLayout = QVBoxLayout()
        lowerLayout.addLayout(mainLayout)


        planificarRutas = QPushButton(self)
        planificarRutas.setText("&Planificar")
        lowerLayout.addWidget(planificarRutas)

        planificarRutas.clicked.connect(self.threadPlanificar)

        #self.connect(self.planificarB, SIGNAL("clicked()"),self.button_click)
    
        self.setLayout(lowerLayout)
        
        self.setWindowTitle("Planificar")
        #self.changeStyle('Windows')

    '''
    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)
    '''
    
    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QTabWidget()
        self.bottomLeftTabWidget.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Ignored)

        '''
        TAB 1 - GENERAL 
        '''
        tab1 = QWidget()
        
        tab1hbox = QFormLayout()
        localidadCombo = QComboBox()
        localidadCombo.addItems(listaLocalidades)

        #localidadEdit = QLineEdit(self)
        localidadLabel = QLabel("&Localidad:", self)
        localidadLabel.setBuddy(localidadCombo)
        tab1hbox.addWidget(localidadLabel)
        #tab1hbox.addWidget(localidadEdit)
        localidadCombo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        tab1hbox.addWidget(localidadCombo)
        
        numDiasEdit = QLineEdit(self)
        numDiasLabel = QLabel("&Número de dias:", self)
        numDiasLabel.setBuddy(numDiasEdit)
        tab1hbox.addWidget(numDiasLabel)
        tab1hbox.addWidget(numDiasEdit)

        capacidadContenedorEdit = QLineEdit(self)
        capacidadContenedorLabel = QLabel("Capacidad máxima de los contenedores:", self)
        capacidadContenedorLabel.setBuddy(capacidadContenedorEdit)
        tab1hbox.addWidget(capacidadContenedorLabel)
        tab1hbox.addWidget(capacidadContenedorEdit)

        iteracionesEdit = QLineEdit(self)
        iteracionesEdit.setText("5")
        iteracionesLabel = QLabel("Número de iteraciones a realizar:", self)
        iteracionesLabel.setBuddy(iteracionesEdit)
        tab1hbox.addWidget(iteracionesLabel)
        tab1hbox.addWidget(iteracionesEdit)

        vecinadrioEdit = QLineEdit(self)
        vecindarioLabel = QLabel("Tamaño del vecindario:", self)
        vecindarioLabel.setBuddy(vecinadrioEdit)
        tab1hbox.addWidget(vecindarioLabel)
        tab1hbox.addWidget(vecinadrioEdit)

        guardadGeneral = QPushButton(self)
        guardadGeneral.setText("Guardar")
        tab1hbox.addWidget(guardadGeneral)
        guardadGeneral.clicked.connect(lambda checked, obj=[localidadCombo,numDiasEdit,capacidadContenedorEdit,iteracionesEdit,vecinadrioEdit] : self.guardarDatos(obj))

        tab1.setLayout(tab1hbox)


        '''
        TAB 2 - TABLA CON INFORMACIÓN DE CAMIONES 
        '''

        
        tab2 = QWidget()
        camionesTableWidget = QTableWidget(10, 10)

        tab2hbox = QGridLayout()
        tab2hbox.setContentsMargins(10, 10, 10, 10)
    
        camionesTableWidget=QTableWidget()
        camionesTableWidget.setColumnCount(len(headers))
        camionesTableWidget.setRowCount(len(datos))

        j = 0
        for h in headers: 
            camionesTableWidget.setHorizontalHeaderItem(j,QTableWidgetItem(h))
            j += 1

        i = 0
        while i < len(datos): 
            cont = 0
            while cont < len(datos[0]): 
                #no pilla el último

                camionesTableWidget.setItem(i,cont,QTableWidgetItem(str(datos[i][cont])))
                #Set icon (para la última columna ?)
                cont += 1

            i += 1 

        def guardarCamiones(self):
            r = camionesTableWidget.rowCount()
            c = camionesTableWidget.columnCount()
            print(r)
            print(c)
            Matrix = [[0 for x in range(c)] for y in range(r)]    
            print(Matrix)
            row = 0
            col = 0
            for i in range(camionesTableWidget.columnCount()):
                for x in range(camionesTableWidget.rowCount()):
                    try:
                        text = str(camionesTableWidget.item(row, col).text())
                        Matrix[x][i] = int(text)
                        #datos[x][i] = int(text)
                        
                        row += 1
                    except AttributeError:
                        row += 1
                row = 0
                col += 1
            print("Datos Camiones")
            datos = Matrix
            print(datos)
            with open("Data/Camiones.csv", "w", newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(headers) # write the header
                # write the actual content line by line
                for d in datos:
                    writer.writerow(d)

        def añadirCamiones(self): 
          rowPosition = camionesTableWidget.rowCount()
          camionesTableWidget.insertRow(rowPosition)

        def borrarFila(self):
          camionesTableWidget.removeRow(camionesTableWidget.currentRow())



        añadirCamionesB= QPushButton(self)
        añadirCamionesB.setText("Añadir fila")
        tab2hbox.addWidget(añadirCamionesB,0,1)
        añadirCamionesB.clicked.connect(añadirCamiones)


        borrarFilaB= QPushButton(self)
        borrarFilaB.setText("Eliminar fila")
        tab2hbox.addWidget(borrarFilaB,1,1)
        borrarFilaB.clicked.connect(borrarFila)

        tab2hbox.addWidget(camionesTableWidget,2,0)

        guardarCamionesB= QPushButton(self)
        guardarCamionesB.setText("Guardar")
        tab2hbox.addWidget(guardarCamionesB,3,0)
        guardarCamionesB.clicked.connect(guardarCamiones)




        tab2.setLayout(tab2hbox)

   
        '''
        TAB 3 - TABLA CON INFORMACIÓN DE CONTENEDORES 
        '''

        tab3 = QWidget()
        contenedoresTableWidget = QTableWidget(10, 10)

        tab3hbox = QHBoxLayout()
        tab3hbox.setContentsMargins(5, 5, 5, 5)
      
        contenedoresTableWidget=QTableWidget()
        contenedoresTableWidget.setColumnCount(len(headersContenedores))
        contenedoresTableWidget.setRowCount(len(datosContenedores))


        j = 0
        for h in headersContenedores: 
            contenedoresTableWidget.setHorizontalHeaderItem(j,QTableWidgetItem(h))
            j += 1

        i = 0
        while i < len(datosContenedores): 
            cont = 0
            while cont < len(datosContenedores[0]): 
                #no pilla el último
                contenedoresTableWidget.setItem(i,cont,QTableWidgetItem(str(datosContenedores[i][cont])))
                #Set icon (para la última columna ?)
                cont += 1

            i += 1 

        def guardarContenedores(self):
        
            row = 0
            col = 0
            for i in range(contenedoresTableWidget.columnCount()):
                for x in range(contenedoresTableWidget.rowCount()):
                    try:
                        text = str(contenedoresTableWidget.item(row, col).text())
                        datosContenedores[x][i] = (text)
                        row += 1
                    except AttributeError:
                        row += 1
                row = 0
                col += 1

            with open("Data/Contenedores.csv", "w", newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(headers) # write the header
                # write the actual content line by line

                for d in datosContenedores:
                    writer.writerow(d)


        tab3hbox.addWidget(contenedoresTableWidget)
        guardarContenedoresB = QPushButton(self)
        guardarContenedoresB.setText("Guardar")
        tab3hbox.addWidget(guardarContenedoresB)
        guardarContenedoresB.clicked.connect(guardarContenedores)
        tab3.setLayout(tab3hbox)

        '''
        TAB 4 - MUESTRA RUTAS 
        '''
        tab4 = QWidget()

        tabMapas = QTabWidget()

        i = 0

        for archivo in os.listdir('Resultados/'):
          if(archivo.endswith('html')):
            i += 1
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Resultados/mapa{0}.html".format(i)))
            local_url = QUrl.fromLocalFile(file_path)
            browser = QWebEngineView()
            browser.load(local_url)

            tabMapa = QWidget()
            tabMapabox = QHBoxLayout()
            tabMapabox.setContentsMargins(5, 5, 5, 5)
            tabMapabox.addWidget(browser)
            tabMapa.setLayout(tabMapabox)
            tabMapas.addTab(tabMapa, "Dia {}".format(i))

        #tab4hbox = QHBoxLayout()
        tab4hbox = QHBoxLayout()
        tab4hbox.setContentsMargins(5, 5, 5, 5)
        tab4hbox.addWidget(tabMapas)
        tab4.setLayout(tab4hbox)


        #tab4hbox.addWidget(browser)
        #browser.show() #Si descomentamos esto se abre y se cierra una ventana antes de que salga la ventana de la aplicación.

        
        '''
        TAB 5 - Resultados 
        '''
        tab5 = QWidget()
        tab5hbox = QHBoxLayout()
        tab5hbox.setContentsMargins(5, 5, 5, 5)
        text_edit = QPlainTextEdit()

        text = open('Data/sample.txt').read()
        text_edit.setReadOnly(True)
        text_edit.appendPlainText(text)
        tab5hbox.addWidget(text_edit)
        tab5.setLayout(tab5hbox)

        self.bottomLeftTabWidget.addTab(tab1, "&General")
        self.bottomLeftTabWidget.addTab(tab2, "&Camiones")
        self.bottomLeftTabWidget.addTab(tab3, "&Contenedores")
        self.bottomLeftTabWidget.addTab(tab4, "&Plan")
        self.bottomLeftTabWidget.addTab(tab5, "&Resultados")

    def guardarDatos(self,obj):
        # shost is a QString object
        datosPlanificar['Localidad'] = str(obj[0].currentText())
        datosPlanificar['numDias'] = obj[1].text()
        datosPlanificar['capacidadContenedor'] = obj[2].text()
        datosPlanificar['iteraciones'] = obj[3].text()
        datosPlanificar['vecindario'] = obj[4].text()

    def planificar(self): 
      for dc in datosContenedores: 
          datosPlanificar['estadoInicial'].append(dc[1])
          datosPlanificar['aumentoDiario'].append(dc[2])

      datos = self.leerCSV('Camiones.csv', headers)
      print('datos en planificar:', datos)

      for c in datos: 
          if c[3] == 1: 
              datosPlanificar['numCamiones']  += 1
              datosPlanificar['capacidadCamiones'].append(c[1])
              datosPlanificar['velocidadCamiones'].append(c[2])


      ''' 
      VARIABLES 
      '''
      localidad = datosPlanificar['Localidad']
      capacidadCamiones = datosPlanificar['capacidadCamiones']
      nCamiones = int(datosPlanificar['numCamiones'])
      depot = 0
      numDias = int(datosPlanificar['numDias'])
      llenadoInicial = datosPlanificar['estadoInicial']
      aumentoDiario = datosPlanificar['aumentoDiario']
      separadorV = ","
      capacidadContenedor = int(datosPlanificar['capacidadContenedor'])
      iteraciones = int(datosPlanificar['iteraciones'])
      tamanyoVecindario = int(datosPlanificar['vecindario'])

      print(localidad)
      import time 
      start_time = time.time()

      """### Main

      Comentario: Introduciendo los parametros de esta forma no tenemos la opción de variar el estado inical o el aumento de cada contenedor de forma individual. Revisar.
      """
      #localidad = 'ABADINO'
      #nCamiones = 5
      #capacidadCamiones = 700
    

      capacidadCamiones = fromCharToInt(procesaVector(capacidadCamiones,separadorV))
      data = create_data_model2(localidad, capacidadCamiones, nCamiones, depot, capacidadContenedor)
      
      ncontenedores = len(data['distance_matrix']) 
      llenadoInicial = llenadoInicial[0:ncontenedores]
      aumentoDiario = aumentoDiario[0:ncontenedores]

      i = 0 
      capacidadTotal = 0
      while i < nCamiones:      
          capacidadTotal += capacidadCamiones[i]
          i += 1 

      #numDias = 3 # valor máximo del plan

      nCont = len(data['datos'])
      #estadoContenedores, aumentoDiario = init(nCont)
      estadoContenedores = pd.DataFrame(fromCharToInt(procesaVector(llenadoInicial, separadorV)))
      aumentoDiario = pd.DataFrame(fromCharToInt(procesaVector(aumentoDiario, separadorV)))
      plan = randomPlan(nCont, numDias)

      #plan = sacarPlan(data, capacidadContenedor, numDias, capacidadTotal, estadoContenedores, aumentoDiario)
      #en algún punto... quitar los 5 minutos en la time-matrix y ponerlos como "de servicio"

      # costes de este plan 
      costesIniciales, results = funcionCostes(data, plan, estadoContenedores, aumentoDiario, capacidadTotal, capacidadContenedor)

      costeInicial = 0 
      for c in costesIniciales: 
          costeInicial += c; 

      print("\n######################")
      print("costeInicial: ", costeInicial)
      print("planInicial: ", dfToList(plan))
      print("######################\n")

      plan, costes = optimizacion(plan, costeInicial, ncontenedores, estadoContenedores, aumentoDiario, data, iteraciones,  capacidadTotal, tamanyoVecindario, capacidadContenedor)
      print("\nDespues de la optimización")
      print("\n######################")
      print("plan: ", dfToList(plan))
      print("costes: ", costes)
      print("######################\n")

      coste, resultado, demandas = funcion(data, plan, estadoContenedores, aumentoDiario, capacidadTotal, localidad, capacidadContenedor)
      lat, longi, depot = getCoordenadas(data)

      '''
      print("\n\nCÓDIGO DE COLORES")
      print("- - - - - - - - - -")
      print("Azul - correctas")
      print("Amarillo - límite")
      print("Rojo - desbordadas")
      '''
      
      print("\nCostes: ", coste)

      print("--- %s seconds ---" % (time.time() - start_time)) 

      d = 0
      #try:
      #while d < numDias:  
      while d < len(resultado):
          listaR = []
          listaT = []
          listaL = []
          listaLC = []
          ncam = 0
         
          while ncam < nCamiones: 
          # sale index out of range
            listaR.append(resultado[d]['listaRutas'][ncam])
            listaT.append(resultado[d]['listaTiempos'][ncam])
            listaL.append(resultado[d]['listaCargas'][ncam])#Es el total llenado del camión
            listaLC.append(resultado[d]['listaLlenadoC'][ncam])
            ncam +=1
          
          #print(listaR)
          mapas = get_map(lat, longi, depot, capacidadCamiones, capacidadContenedor, listaR, listaT, listaL, listaLC)
          for mapa in mapas:
            mapa.save("Resultados/mapa"+str(d+1)+".html")
          
          d += 1

      #except:
          #print("Las rutas del día {} hace que se desborden contenedores. Se ha dejado de planificar.".format(d+1))     

    def threadPlanificar(self):
      self._planThread.start()

    def leerCSV(self, nombre, headers):
      data = pd.read_csv('Data/'+nombre, delimiter=',', header=0, names=headers)
      datos = data.values.tolist()
      return datos

    def closeEvent(self, event):
      event.accept()


if __name__ == '__main__':

    import sys
   

    app = QApplication(sys.argv)
    
    gallery = WidgetGallery()
    
    gallery.show()

    
    sys.exit(app.exec_()) 


