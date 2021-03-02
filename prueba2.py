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


class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

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
   
        self.setLayout(mainLayout)
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
        

        tab1.setLayout(tab1hbox)

        tab2 = QWidget()
        tableWidget = QTableWidget(10, 10)

        tab2hbox = QHBoxLayout()
        tab2hbox.setContentsMargins(5, 5, 5, 5)
      

        tableWidget=QTableWidget()
        tableWidget.setColumnCount(len(headers))
        tableWidget.setRowCount(len(datos)+1)

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

        tableWidget.doubleClicked.connect(getSelectedItemData)
        tab2hbox.addWidget(tableWidget)
        tab2.setLayout(tab2hbox)

        tab3 = QWidget()
        tab3hbox = QHBoxLayout()
        tab3hbox.setContentsMargins(5, 5, 5, 5)
       # tab3hbox.addWidget()
        tab3.setLayout(tab3hbox)

        tab4 = QWidget()
        tab4hbox = QHBoxLayout()
        tab4hbox.setContentsMargins(5, 5, 5, 5)
       # tab4hbox.addWidget()
        tab4.setLayout(tab4hbox)

        self.bottomLeftTabWidget.addTab(tab1, "&General")
        self.bottomLeftTabWidget.addTab(tab2, "&Camiones")
        self.bottomLeftTabWidget.addTab(tab3, "&Contenedores")
        self.bottomLeftTabWidget.addTab(tab4, "&Plan")
   


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    headers = ['Camión','Capacidad','Velocidad','Funcionando']

    data = pd.read_csv('test.csv', delimiter=',', header=0, names=headers)

    datos = data.values.tolist() 
    print(datos)
    
    gallery = WidgetGallery()
    
    gallery.show()
    
    sys.exit(app.exec_()) 