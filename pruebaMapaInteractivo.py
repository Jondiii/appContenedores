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

    return latitud, longitud

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

def get_map(localidad, rutas):
    datos = leerDatos(localidad)
    lat, longi = getCoordenadas(datos)
    mapas = []

    for ruta in rutas:
        out = get_route(ruta, lat, longi)

        m = folium.Map(location=[(out['depot'][0] + out['depot'][0])/2, 
                                (out['depot'][1] + out['depot'][1])/2], 
                    zoom_start=13)

        folium.PolyLine(
            out['route'],
            weight=8,
            color='blue',
            opacity=0.6
        ).add_to(m)

        folium.Marker(
            location=out['depot'],
            icon=folium.Icon(icon='play', color='green')
        ).add_to(m)

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
            ).add_to(m)

        mapas.append(m)

    return mapas

    
localidad = 'ABADINO'

mapas = get_map(localidad, listaRutas2)
#https://www.thinkdatascience.com/post/2020-03-03-osrm/osrm/

i = 0
for mapa in mapas:
    i += 1
    mapa.save(localidad+" - dia "+str(i)+".html")

#De momento solo crea un html en el directorio local.
