from __future__ import print_function

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
import folium
import polyline
import random

#Interesting links with examples 
#https://realpython.com/pysimplegui-python/
#https://pysimplegui.readthedocs.io/en/latest/cookbook/ #Este tiene también sobre los colores.
#https://pypi.org/project/PySimpleGUI/


#sg.Window(title="Planificador de rutas", layout= [[sg.Text("Planificador de rutas")], [sg.Button("OK")]], margins=(500, 200)).read()



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

def create_data_model2(localidad, capacidadCamiones, ncamiones, depot):

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

"""#### Métodos matriz de distancias
"""

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

El siguiente método creará el Routing Model. Para ello, primero se tiene que crear el Index Manager, los cuales se utilizan para señalizar los nodos por los que se estén pasando. Primero se pasa el número de contenedores, luego los vehículos y finalmente el punto de partida.

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

def getCoordenadas(datos):
    longitud = datos['longitude'].to_numpy()
    latitud = datos['latitude'].to_numpy()
    #Transformamos de UTM a Coordenadas Geográficas (Grados)
    scrProj = pyproj.Proj(proj="utm", zone = 30, ellps="WGS84", units = "m")
    dstProj = pyproj.Proj(proj = "longlat", ellps="WGS84", datum = "WGS84")

    i = 0
    for n in datos['latitude']:
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
  # DUDA converitri cada elemento en un integer ? 
  return vector.split(separadorV)

def fromCharToString(vector):
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

def sacarPlan(data, sizeCont, nDias, capacidadTotal):

  estadoI = [0]
  aumentoD = [0]
  print("total truck capacity: ", capacidadTotal)
  plan = [0]
  
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

      elif (((recogido + n) <= capacidadTotal) & (n > 20) & (plan[cont]==nDias+1)):#Dejamos los que tengan menos de 20% para "rellenar" posibles huecos
        recogido = recogido + n
        newEstado[0][cont] = 0 #Se ha recogido el contenedor.
        plan[cont] = i
       
      cont += 1
    
    cont = 0
    for n in newEstado[0]:
      if (((recogido + n) <= capacidadTotal) & (n <= 200) & (n != 0) & (plan[cont]==nDias+1)):
        recogido = recogido + n
        newEstado[0][cont] = 0
        plan[cont] = i
      cont =+ 1
    

    estadoDF = newEstado.copy() + aumentoDF
    i += 1  

  plan = pd.DataFrame(plan)
  #print("plan: ", dfToList(plan))

  return pd.DataFrame(estadoI), aumentoDF, plan

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
    resultado = {}

    for vehicle_id in range(data['num_vehicles']):
      index = routing.Start(vehicle_id)
      route_distance = 0
      route_load = 0
      route_time = 0
      ruta = []
      tiempos = []
      cargas = []

      while not routing.IsEnd(index):
          node_index = manager.IndexToNode(index)
          route_load += data['demands'][node_index]
          
          #Guardamos datos de las rutas
          ruta.append(node_index)
          tiempos.append(route_time)
          cargas.append(toFloat(route_load))

          #Para la distancia
          previous_index = index
          index = solution.Value(routing.NextVar(index))
          route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
          
          #Para el timempo
          previous_node_index = node_index
          node_index = manager.IndexToNode(index)
          route_time += data['time_matrix'][previous_node_index][node_index];

      ruta.append(data['depot'])
      tiempos.append(route_time)
      cargas.append(toFloat(route_load))

      listaRutas.append(ruta)
      listaTiempos.append(tiempos)
      listaCargas.append(cargas)

      total_distance += route_distance
      total_load += route_load
      total_time += route_time
    
    resultado['listaRutas'] = listaRutas
    resultado['listaTiempos'] = listaTiempos
    resultado['listaCargas'] = listaCargas
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
    title = ("Rutas del día %i" % (d)) 
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

def get_map(datos, rutas):
  lat, longi, depot = getCoordenadas(datos)
  mapas = []

  m = folium.Map(location=depot,
              zoom_start=13)

  n = 0
  for ruta in rutas:
      n += 1
      out = get_route(ruta, lat, longi)
      feature_group = folium.FeatureGroup(name="Ruta "+str(n))

      folium.PolyLine(
          out['route'],
          weight=8,
          color=random.choice(('blue', 'red', 'green', 'yellow', 'black', 'purple', 'orange', 'brown', 'gray')),
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
      for point in out['ruta']:
          i += 1
          folium.Marker(
              locationList[point], tooltip=str(i-1)
          ).add_to(feature_group)

      
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

def funcion(data, plan, estadoContenedores, aumentoDiario, capacidadTotal):
  if (len(data['datos'])!=len(plan)):
    raise Exception("El número de contenedores en el plan y en el data no coinciden")

  i = 0
  # print("max1: ",plan.max()[0])
  #print(plan.max())

  numDias = int(plan.max())
  results = []
  costes = []
  dataOriginal = data
  #capacidadTotal = data['num_vehicles'] * data['vehicle_capacities'][0]
  demandas = []

  print("Estado inicial: ", dfToList(estadoContenedores))
  print("#Cantidad de toneladas que tienen los contenedores antes de empezar a recoger\n")

  while i <= numDias:
    # tantos "Números altos" como días vayamos a planificar (+1)
    costes.append(10000)
    i += 1
  
  dia = 1
  print("- - - - - - - - - - - - - - - - - -  ")
  print("PLANIFICACIÓN de ", localidad)
  print("- - - - - - - - - - - - - - - - - -  ")

  while dia <= numDias:
    print("\nDIA: ", dia)
    print("- - - - - - - ")
    indicesRecoger = []

    indices = []
    
    plan[0][0] = dia
    contenedoresARecoger = plan[plan == dia]

    contenedoresARecoger =  (contenedoresARecoger.replace(np.nan, 0))/dia
    demanda = contenedoresARecoger*(estadoContenedores)
    #data['demands'] = demanda

    print("Cont a recoger: ", dfToList(contenedoresARecoger))
    
    contador = int(0)
    for c in dfToList(contenedoresARecoger): 
      if (c == 1): 
        indices.append(contador)
      contador  += 1
  
    print("#Indice de los contenedores a recoger el día" , dia)
    print("Indices:" , indices)
    
    demandaActual = []
    #print("Demanda: ", dfToList(demanda))
    for d in dfToList(demanda): 
      if (d != 0): 
        demandaActual.append(d)
    
    print("Demanda: ", (demandaActual))

    cont = 0
    for i in plan:
      if  i == dia:
        indicesRecoger.append(cont)
      cont += 1

    #print("Indice: ", indicesRecoger)

    valorContenedores = demanda

    data = newDatamodel(dataOriginal, contenedoresARecoger)

    if (demanda[demanda>100].isnull().sum().sum()!=len(demanda)): 
      #Salimos del bucle, dejar de planificar el resto de días.
      desborde = list(np.where(demanda>100)[0])
      print("Contenedor(es) desbordado(s):", desborde)
      print("#contenedor desborda cuando demanda > 100")
      
      break

    else:

      solucion, manager, routing, resultado = solucionaProblema(data)
  
      total = 0
      for n in data['demands']:
        total = total + n
      #print("Total demandas: ", np.sum(demanda))
      
      if solucion:
        costes[dia-1] = -10000 * (resultado['total_distance']/1000)  
        # / 1000 para pasar a km 
      
        estadoContenedores = estadoContenedores * (1-contenedoresARecoger)
       
        estadoContenedores = estadoContenedores + aumentoDiario
        

        results.append(resultado)
        
        #print("  Solución encontrada. Resultado: ", resultado)
        nuevasRutas = []
        print("Rutas: ")
        #Solo imprime las rutas que pasen 
        for element in resultado['listaRutas']:
          nuevasRutas.append([indices[i] for i in element])
          if (len(element) > 2):
            print([indices[i] for i in element])
            
        print("\n - Distancia Total: ", resultado['total_distance'], "m")
        print(" - Carga Total: ", resultado['total_load'], "toneladas")
        print(" - Tiempo Total: ", resultado['total_time'], "min")
        #print(nuevasRutas)
        resultado['listaRutas'] = nuevasRutas
  
      else:  # Si el plan para ese día no es válido, no se puede recoger con los camiones que tenemos
        costes[dia-1] = int(costes[dia-1]+estadoContenedores.sum())
        estadoContenedores = estadoContenedores + aumentoDiario
        print("  No tengo suficientes camiones: ", costes[dia-1])
        
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

  return costes, results, demandas

"""####FuncionCostes"""

def funcionCostes(data, plan, estadoContenedores, aumentoDiario, capacidadTotal):
  if (len(data['datos'])!=len(plan)):
    raise Exception("El número de contenedores en el plan y en el data no coinciden")

  i = 0
  numDias = plan.max()[0]
  results = []
  costes = []
  dataOriginal = data

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

    valorContenedores = demanda

    data = newDatamodel(dataOriginal, contenedoresARecoger)

    if (demanda[demanda>100].isnull().sum().sum()!=len(demanda)): 
      #Salimos del bucle, dejar de planificar el resto de días.
      desborde = list(np.where(demanda>100)[0])

      for d in desborde:
        costes[dia-1] = costes[dia-1]+dfToList(estadoContenedores)[d]

      #print("  Contenedor(es) desbordado(s) día {0}: {1} - coste: {2}".format(dia, desborde, costes[dia-1]))

      break

    else:
      solucion, manager, routing, resultado = solucionaProblema(data)

      if solucion:
        costes[dia-1] = -10000 * (resultado['total_distance']/1000)  
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

def optimizacion(planInicial, costeInicial, ncontenedores, estadoContenedores, aumentoDiario, data, iteraciones, capacidadTotal, imprime = True): 
  plan = []
  resultados = []
  planes = []
  listaCostes = []
  
  i = 1
  while (i <= iteraciones):

    n = 1
    while (n < ncontenedores-1):
      #if imprime:
        #print("\n·············")
        #print("Iteración: ", i)
        #print("Vecindario: {0}/{1}".format(n, ncontenedores-2)) 

      #plan = suma(planInicial, n, planInicial.max()[0])
      plan = suma2(planInicial, ncontenedores, planInicial.max()[0])
      costes, results = funcionCostes(data, plan, estadoContenedores, aumentoDiario, capacidadTotal)
    
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

listaLocalidades = pd.read_csv("https://raw.githubusercontent.com/Jondiii/appContenedores/master/localidades.txt",delimiter=sep, header=header)




'''
------------------------------------
          V E N T A N A   
------------------------------------
'''

#Mirar si se puede buscador y lista
plan =  [[sg.Text('Número de días', size=(15, 1)), sg.InputText()],
          [sg.Text('Localidad',size=(15, 1)), sg.InputText()]]
   
     
camiones = [[sg.Text('Número de camiones', size=(20, 1)), sg.InputText()],
            [sg.Text('Capacidad de camiones', size=(20, 1)), sg.InputText()],
            [sg.Text('Velocidad de camiones', size=(20, 1)), sg.InputText()]]       

demandas =  [[sg.Text('Llenado inicial', size=(20, 1)), sg.InputText()],
                [sg.Text('Aumento diario', size=(20, 1)), sg.InputText()],
                [sg.Text('Capacidad contenedor', size=(20, 1)), sg.InputText()]]    

dia1 = [[sg.Text('Por ahora nada', size=(15, 1))]]
dia2 = [[sg.Text('Por ahora nada pero en rojo', size=(25, 1), text_color="red")]]


visualizacion = [[sg.TabGroup([[sg.Tab('Día 1', dia1, tooltip='tip'),
                sg.Tab('Día 2', dia2)]], tooltip='TIP2')]]

layout = [[sg.TabGroup([[sg.Tab('Plan', plan, tooltip='tip'),
                sg.Tab('Camiones', camiones, tooltip='TIP2'),
                sg.Tab('Demandas', demandas, tooltip='TIP3'),
                sg.Tab('Visualización', visualizacion)]], tooltip='TIP4')],#Tabs dentro de tab
            [sg.Button('Planificar'), sg.Button('Exit')]]

window = sg.Window('Planificador de rutas', layout, grab_anywhere=False)

while True:  # Event Loop
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    
    if event == 'Planificar': 
 
        numDias = int(values[0])
        localidad = values[1]   
        nCamiones = int(values[2])
        capacidadCamiones = (values[3])
        velCamiones = int(values[4])
        llenadoInicial = values[5]
        aumentoDiario = values[6]
        separadorV = ","
        capacidadContenedor = int(values[7])

        

        print(localidad, nCamiones, capacidadCamiones,  velCamiones, llenadoInicial, aumentoDiario, numDias)    # the input data looks like a simple list when auto numbered
        """### Main

        Comentario: Introduciendo los parametros de esta forma no tenemos la opción de variar el estado inical o el aumento de cada contenedor de forma individual. Revisar.
        """
        #localidad = 'ABADINO'
        #nCamiones = 5
        #capacidadCamiones = 700
        depot = 0

      
        capacidadCamiones = fromCharToString(procesaVector(capacidadCamiones,separadorV))
        data = create_data_model2(localidad, capacidadCamiones, nCamiones, depot)

        ncontenedores = len(data['distance_matrix']) 

        i = 0 
        capacidadTotal = 0
        while i < data['num_vehicles']:      
            capacidadTotal += data['vehicle_capacities'][i]
            i += 1 

        #numDias = 3 # valor máximo del plan

        nCont = len(data['datos'])
        #estadoContenedores, aumentoDiario = init(nCont)
        estadoContenedores = pd.DataFrame(fromCharToString(procesaVector(llenadoInicial, separadorV)))
        aumentoDiario = pd.DataFrame(fromCharToString(procesaVector(aumentoDiario, separadorV)))
        plan = randomPlan(nCont, numDias)

        #en algún punto... quitar los 5 minutos en la time-matrix y ponerlos como "de servicio"

        # costes de este plan 
        costesIniciales, results = funcionCostes(data, plan, estadoContenedores, aumentoDiario, capacidadTotal)

        costeInicial = 0 
        for c in costesIniciales: 
            costeInicial += c; 

        print("\n######################")
        print("costeInicial: ", costeInicial)
        print("planInicial: ", dfToList(plan))
        print("######################\n")

        plan, costes = optimizacion(plan, costeInicial, ncontenedores, estadoContenedores, aumentoDiario, data, 5,  capacidadTotal, imprime = True)
        print("\nDespues de la optimización")
        print("\n######################")
        print("plan: ", dfToList(plan))
        print("costes: ", costes)
        print("######################\n")


        print("SOLUCIÓN")

        coste, resultado, demandas = funcion(data, plan, estadoContenedores, aumentoDiario, capacidadTotal)
        print("\n\nCÓDIGO DE COLORES")
        print("- - - - - - - - - -")
        print("Azul - correctas")
        print("Amarillo - límite")
        print("Rojo - desbordadas")

        #print("resultado: ", resultado)

        d = 0
        try:
            while d < numDias:  
                listaR = []
                ncam = 0

                while ncam < nCamiones: 
                # sale index out of range
            
                    listaR.append(resultado[d]['listaRutas'][ncam])
                    #representarContenedores(listaR, data, localidad)
                    ncam +=1

                representarContenedores(listaR,data,localidad, d, resultado, demandas)
                #print("PUEDE QUE NO SE ESTÉN REPRESENTANDO LAS DEMANDAS CORRECTAMENTE EN EL GRÁFICO.")
                d += 1

        except:
            print("Las rutas del día {} hace que se desborden contenedores. Se ha dejado de planificar.".format(d+1))

        print("\nCostes: ", coste)      

window.close()