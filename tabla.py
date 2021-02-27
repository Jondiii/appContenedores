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
import random
import string
import urllib, json

# No funciona pero he creado este file apra ir haciendo pruebas de la tabla para no estropear la otra venta
# pongo links a un git con ejemplos de tablas: 
# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_CSV.py
# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_Simulation.py
# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_Pandas.py


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

headers = ['Camión','Capacidad','Velocidad','Funcionando']


data = pd.read_csv('https://raw.githubusercontent.com/Jondiii/appContenedores/master/file.csv', delimiter=',', header=0, names=headers)
datos = data.values.tolist() 
print(datos)

layout = [[sg.Table(key='-TABLE-', values=datos,
                        headings=headers,
                        max_col_width=25,
                        auto_size_columns=True,
                        justification='right',
                        # alternating_row_color='lightblue',
                        num_rows=min(len(data), 20))], 
                [sg.Button('Delete'), sg.Button('Add')]] 



window = sg.Window('Table', layout, grab_anywhere=False)
event, values = window.read()

window.close()