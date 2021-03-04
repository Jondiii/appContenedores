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
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QTableWidgetItem)

#https://github.com/pyqt/examples/tree/_/src/02%20PyQt%20Widgets
#https://stackoverflow.com/questions/52010524/widgets-placement-in-tabs
#https://realpython.com/python-pyqt-gui-calculator/


localidad = ""
numDias = 0
capacidadContenedor = 0  

headers = ['Camión','Capacidad','Velocidad','Funcionando']
data = pd.read_csv('test.csv', delimiter=',', header=0, names=headers)
datos = data.values.tolist() 

#print(datos)

headersContenedores = ["ID Contenedor", "Estado Inicial", "Aumento Diario"]
dataContenedores = pd.read_csv('Contenedores.csv', delimiter=',', header=0, names=headersContenedores)
datosContenedores = dataContenedores.values.tolist() 
#print(datosContenedores)


#estos valores se tiene que asignar en el tab general 
localidad = ""
numDias = 3
capacidadContenedor = 0


class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)
        
        self._dataCamiones = datos
        self._dataContenedores = datosContenedores


        self.originalPalette = QApplication.palette()
        
        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)

    
        self.createBottomLeftTabWidget()
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addWidget(self.bottomLeftTabWidget)
        
        lowerLayout = QVBoxLayout()
        lowerLayout.addLayout(mainLayout)


        planificarRutas = QPushButton(self)
        planificarRutas.setText("&Planificar")
        lowerLayout.addWidget(planificarRutas)

        #planificarRutas.clicked.connect(lambda checked,  self.guardarDatos(obj))

        #self.connect(self.planificarB, SIGNAL("clicked()"),self.button_click)
    
        self.setLayout(lowerLayout)
        
        self.setWindowTitle("Styles")
        self.changeStyle('Windows')

       

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    
    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QTabWidget()
        self.bottomLeftTabWidget.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Ignored)

        '''
        TAB 1 - GENERAL 
        '''
        tab1 = QWidget()
        
        tab1hbox = QVBoxLayout()
        
        localidadEdit = QLineEdit(self)
        localidadLabel = QLabel("&Localidad:", self)
        localidadLabel.setBuddy(localidadEdit)
        tab1hbox.addWidget(localidadLabel)
        tab1hbox.addWidget(localidadEdit)
        
        numDiasEdit = QLineEdit(self)
        numDiasLabel = QLabel("&Número de días:", self)
        numDiasLabel.setBuddy(numDiasEdit)
        tab1hbox.addWidget(numDiasLabel)
        tab1hbox.addWidget(numDiasEdit)

        capacidadContenedorEdit = QLineEdit(self)
        capacidadContenedorLabel = QLabel("Capacidad máxima de los contenedores:", self)
        capacidadContenedorLabel.setBuddy(capacidadContenedorEdit)
        tab1hbox.addWidget(capacidadContenedorLabel)
        tab1hbox.addWidget(capacidadContenedorEdit)

        guardadGeneral = QPushButton(self)
        guardadGeneral.setText("Guardar")
        tab1hbox.addWidget(guardadGeneral)
        guardadGeneral.clicked.connect(lambda checked, obj=[localidadEdit,numDiasEdit,capacidadContenedorEdit] : self.guardarDatos(obj))

        tab1.setLayout(tab1hbox)


        '''
        TAB 2 - TABLA CON INFORMACIÓN DE CAMIONES 
        '''

        tab2 = QWidget()
        camionesTableWidget = QTableWidget(10, 10)

        tab2hbox = QHBoxLayout()
        tab2hbox.setContentsMargins(5, 5, 5, 5)
      

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

        def getSelectedItemData():
            for currentItem in camionesTableWidget.selectedItems():
                print("Row : "+str(currentItem.row())+" Column : "+str(currentItem.column())+" "+currentItem.text())
                #falta algún tipo de refresh que nos permita guardar los cambios
                datos[currentItem.row()][currentItem.column()] = int(currentItem.text())
                
                
                with open("test.csv", "w", newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(headers) # write the header
                    # write the actual content line by line
                    for d in datos:
                        writer.writerow(d)
                    


        camionesTableWidget.clicked.connect(getSelectedItemData)
        tab2hbox.addWidget(camionesTableWidget)
        guardarCamiones = QPushButton(self)
        guardarCamiones.setText("Guardar")
        tab2hbox.addWidget(guardarCamiones)
        guardarCamiones.clicked.connect(lambda checked, obj=camionesTableWidget : self.guardarCamiones(obj))



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
        contenedoresTableWidget.setRowCount(len(datosContenedores)+1)

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

        def getSelectedItemData2():
            for currentItem in contenedoresTableWidget.selectedItems():
                print("Row : "+str(currentItem.row())+" Column : "+str(currentItem.column())+" "+currentItem.text())
                
                datos[currentItem.row()][currentItem.column()] = int(currentItem.text())
                #data.loc[currentItem.row(), currentItem.column()] = currentItem.text()
                print(datos)
                with open("Contenedores.csv", "w", newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(headers) # write the header
                    # write the actual content line by line
                    for d in datos:
                        writer.writerow(d)

                #datos[currentItem.row()][currentItem.column()] = currentItem.text()

        contenedoresTableWidget.doubleClicked.connect(getSelectedItemData2)
        tab3hbox.addWidget(contenedoresTableWidget)
        tab3.setLayout(tab3hbox)

        '''
        TAB 4 - MUESTRA RUTAS 
        '''

        tab4 = QWidget()
        tab4hbox = QHBoxLayout()
        tab4hbox.setContentsMargins(5, 5, 5, 5)
       # tab4hbox.addWidget()
        tab4.setLayout(tab4hbox)

        self.bottomLeftTabWidget.addTab(tab1, "&General")
        self.bottomLeftTabWidget.addTab(tab2, "&Camiones")
        self.bottomLeftTabWidget.addTab(tab3, "&Contenedores")
        self.bottomLeftTabWidget.addTab(tab4, "&Plan")



    def guardarDatos(self,obj):
        # shost is a QString object
        
        localidad = obj[0].text()
        numDias = obj[1].text()
        capacidadContenedor = obj[2].text()

        print(localidad)
        print(numDias)
        print(capacidadContenedor)

      
       



        
        
        


if __name__ == '__main__':

    import sys
   

    app = QApplication(sys.argv)
     


    ''' 
    LECTURA DE DATOS PARA TABLAS 
    '''



    # somehow update

    #once it is updated 
    ## DATOS CAMIONES 
    numCamiones = 0 
    capacidadCamiones = []
    valocidadCamiones = []
    for c in datos: 
        if c[3] == 1: 
            numCamiones += 1
            capacidadCamiones.append(c[1])
            valocidadCamiones.append(c[2])

    ##DATOS CONTENEDORES
    estadoInicial = datosContenedores[1]
    aumentoDiario = datosContenedores[2]

    print(numCamiones)
    print(capacidadCamiones)
    print(valocidadCamiones)
    print(estadoInicial)

    gallery = WidgetGallery()
    
    gallery.show()

    '''
    SACAR VARIABLES 
    '''

    
   
    
    sys.exit(app.exec_()) 

