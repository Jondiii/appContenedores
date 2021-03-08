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


#Interesting links with examples 
#https://realpython.com/pysimplegui-python/
#https://pysimplegui.readthedocs.io/en/latest/cookbook/ #Este tiene también sobre los colores.
#https://pypi.org/project/PySimpleGUI/

header = 0
nom_col = ["id","UDALERRIA_KODEA/COD_MUNICIPIO","UDALERRIA/MUNICIPIO","EKITALDIA/EJERCICIO","EDUKIONTZI_KODEA/COD_CONTENEDOR","EDUKIONTZIAREN MODELOA_CAS/MODELO CONTENEDOR_CAS","HONDAKINAREN FRAKZIOA_CAS/FRACCION DEL RESIDUO_CAS","longitude","latitude", "horaMin", "horaMax"]
sep = ','

listaLocalidades = pd.read_csv("https://raw.githubusercontent.com/Jondiii/appContenedores/master/localidades.txt",delimiter=sep, header=header)


#Mirar si se puede buscador y lista
plan =  [[sg.Text('Número de días', size=(15, 1)), sg.InputText()],
          [sg.Text('Localidad')],
          [sg.Input(size=(20, 1), enable_events=True, key='-INPUT-')],
          #Lista 
          [sg.Listbox(values = listaLocalidades, size=(100,8),enable_events=True, key='-LIST-')]]
          #No es necesario, es un intento para ver si detecta el valor
          #[sg.Listbox(values=listaLocalidades, size=(100, 8), key='-LIST-', enable_events=True,bind_return_key=True)]]
          #Otra opción qe igual es mejor
          #[sg.Combo(listaLocalidades, size=(15, 1), key='_LIST_')]]
  

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

window = sg.Window('Planificador de rutas', layout, grab_anywhere=False).Finalize()


while True:  # Event Loop
    event, values = window.read()
    localidad = []
  
    
    if values['-INPUT-'] != '':                         # if a keystroke entered in search field
        search = values['-INPUT-']
        new_values = [x for x in listaLocalidades if search in x]  # do the filtering
        window['-LIST-'].update(new_values)    # display original unfiltered list
    else: 
        window['-LIST-'].update(listaLocalidades)
        localidad = values['-LIST-']
    
    if event == 'Planificar': 

      # if a list item is chosen
        #NO CONSIGO GUARDAR EL VALOR
        #EJEMPLOS
        #https://github.com/PySimpleGUI/PySimpleGUI/issues/1633 
        #Pensado para el botón pero no parece funcionar
    
        
       
        numDias = int(values[0])
        #localidad = values[1]
        #localidad = l[0]
        
        print(localidad)
        nCamiones = int(values[1])
        capacidadCamiones = values[2]
        velCamiones = int(values[3])
        llenadoInicial = values[4]
        aumentoDiario = values[5]
        capacidadContenedor = int(values[6])
        print(values)
        

        print(localidad, nCamiones, capacidadCamiones,  velCamiones, llenadoInicial, aumentoDiario, numDias)    # the input data looks like a simple list when auto numbered
  
    if event == sg.WIN_CLOSED or event == 'Exit':
        break

window.close()