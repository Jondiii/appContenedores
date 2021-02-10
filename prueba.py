import PySimpleGUI as sg

#https://realpython.com/pysimplegui-python/
sg.Window(title="Planificador de rutas", layout= [[sg.Text("Planificador de rutas")], [sg.Button("OK")]], margins=(500, 200)).read()