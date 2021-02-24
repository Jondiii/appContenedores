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

# No funciona pero he creado este file apra ir haciendo pruebas de la tabla para no estropear la otra venta
# pongo links a un git con ejemplos de tablas: 
# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_CSV.py
# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_Simulation.py
# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Table_Pandas.py


url = "https://github.com/Jondiii/appContenedores/blob/master/camiones2.json"


r = requests.get(url)
data = pd.DataFrame(r.json())
print (data)

print (data['1'])




headings = ['Camión','Capacidad','Velocidad', 'Funcionando']
#df = pd.DataFrame (data, columns = ['Camión','Capacidad','Veloidad', 'Funcionando'])
#print(df)





layout = [[sg.Table(values=data,
                        headings=headings,
                        max_col_width=50,
                        auto_size_columns=True,
                        justification='right',
                        # alternating_row_color='lightblue',
                        num_rows=min(len(data), 20))]]


window = sg.Window('Table', layout, grab_anywhere=False)
event, values = window.read()

window.close()