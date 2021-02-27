import requests
import folium
import polyline
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


listaRutas1 = [[0, 57, 50, 19, 18, 46, 25, 52, 29, 14, 62, 12, 5, 37, 9, 8, 30, 4, 33, 42, 39, 63, 58, 55, 40, 38, 43, 28, 0]]
listaRutas2 = [[0, 17, 16, 53, 45, 26, 61, 35, 0], [0, 15, 11, 31, 1, 34, 0], [0, 59, 23, 69, 41, 44, 24, 60, 66, 68, 64, 65, 0]]

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

    return datos

def trasnformaCoordenadas(datos):
    longitud = datos['longitude'].to_numpy()
    latitud = datos['latitude'].to_numpy()
    #Transformamos de UTM a Coordenadas Geográficas (Grados)
    scrProj = pyproj.Proj(proj="utm", zone = 30, ellps="WGS84", units = "m")
    dstProj = pyproj.Proj(proj = "longlat", ellps="WGS84", datum = "WGS84")

    i = 0
    for n in datos['latitude']:
        longitud[i],latitud[i]=pyproj.transform(scrProj,dstProj, longitud[i],latitud[i])
        i +=1

    datos['longitude'] = longitud
    datos['latitude'] = latitud

    return datos

def get_route(rutas, lat, longi):

    stringRutas = []
    for ruta in rutas:
        strRuta = ""
        for contenedor in ruta:
            strRuta = strRuta + str(longi[contenedor]) + "," + str(lat[contenedor]) + ";"

        strRuta = strRuta[:-1]
        stringRutas.append(strRuta)
 
    url = 'http://router.project-osrm.org/route/v1/driving/'+stringRutas[0]+'?annotations=distance,duration'
    print(stringRutas[0])
    r = requests.get(url)
    res = r.json()

    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']
    
    out = {'route':routes,
           'start_point':start_point,
           'end_point':end_point,
           'distance':distance
          }

    return out

def get_map(route):
    m = folium.Map(location=[(route['start_point'][0] + route['end_point'][0])/2, 
                             (route['start_point'][1] + route['end_point'][1])/2], 
                   zoom_start=13)

    folium.PolyLine(
        route['route'],
        weight=8,
        color='blue',
        opacity=0.6
    ).add_to(m)

    folium.Marker(
        location=route['start_point'],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)

    folium.Marker(
        location=route['end_point'],
        icon=folium.Icon(icon='stop', color='red')
    ).add_to(m)

    return m

localidad = 'ABADINO'

datos = leerDatos(localidad)
datos = trasnformaCoordenadas(datos)

route = get_route(listaRutas1, datos["latitude"], datos["longitude"])
mapa = get_map(route)
#https://www.thinkdatascience.com/post/2020-03-03-osrm/osrm/

#De momento solo crea un html en el directorio local.
mapa.save(localidad+".html")