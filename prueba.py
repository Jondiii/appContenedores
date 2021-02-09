import PySimpleGUI as sg

#https://realpython.com/pysimplegui-python/
sg.Window(title="Prueba", layout= [[sg.Text("Hello from PySimpleGUI")], [sg.Button("OK")]], margins=(500, 200)).read()