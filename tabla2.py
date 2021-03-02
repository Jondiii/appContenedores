import sys
from PyQt5.QtWidgets import QWidget,QApplication,QTableWidget,QTableWidgetItem,QVBoxLayout
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


headers = ['Camión','Capacidad','Velocidad','Funcionando']

data = pd.read_csv('test.csv', delimiter=',', header=0, names=headers)

datos = data.values.tolist() 
print(datos)

def getSelectedItemData():
    for currentItem in tableWidget.selectedItems():
        print("Row : "+str(currentItem.row())+" Column : "+str(currentItem.column())+" "+currentItem.text())
        
        datos[currentItem.row()][currentItem.column()] = int(currentItem.text())
        #data.loc[currentItem.row(), currentItem.column()] = currentItem.text()
        print(datos)
        with open("test.csv", "w", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(headers) # write the header
            # write the actual content line by line
            for d in datos:
                writer.writerow(d)

        #datos[currentItem.row()][currentItem.column()] = currentItem.text()

app=QApplication(sys.argv)

qwidget=QWidget()

qwidget.setWindowTitle("Python GUI Table")
qwidget.resize(600,400)

layout=QVBoxLayout()

tableWidget=QTableWidget()
tableWidget.setColumnCount(len(headers))
tableWidget.setRowCount(len(datos)+1)

#adding item in table


j = 0
for h in headers: 
    tableWidget.setHorizontalHeaderItem(j,QTableWidgetItem(h))
    j += 1

i = 0
while i < len(datos): 
    cont = 0
    while cont < len(datos[0]): 
        #no pilla el último
        tableWidget.setItem(i,cont,QTableWidgetItem(str(datos[i][cont])))
        #Set icon (para la última columna ?)
        cont += 1

    i += 1 



tableWidget.doubleClicked.connect(getSelectedItemData)
layout.addWidget(tableWidget)
qwidget.setLayout(layout)
app.setStyle("Breeze")
qwidget.show()




sys.exit(app.exec_())

