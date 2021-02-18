import PySimpleGUI as sg

#Interesting links with examples 
#https://realpython.com/pysimplegui-python/
#https://pysimplegui.readthedocs.io/en/latest/cookbook/ #Este tiene también sobre los colores.
#https://pypi.org/project/PySimpleGUI/


#sg.Window(title="Planificador de rutas", layout= [[sg.Text("Planificador de rutas")], [sg.Button("OK")]], margins=(500, 200)).read()

plan =  [[sg.Text('Localidad', size=(15, 1)), sg.InputText()],
                [sg.Text('Número de días', size=(15, 1)), sg.InputText()]]    

camiones = [[sg.Text('Número de camiones', size=(15, 1)), sg.InputText()],
            [sg.Text('Capacidad de camiones', size=(15, 1)), sg.InputText()],
            [sg.Text('Velocidad de camiones', size=(15, 1)), sg.InputText()]]       

demandas =  [[sg.Text('Llenado inicial', size=(15, 1)), sg.InputText()],
                [sg.Text('Aumento diario', size=(15, 1)), sg.InputText()]]    

dia1 = [[sg.Text('Por ahora nada', size=(15, 1))]]
dia2 = [[sg.Text('Por ahora nada pero en rojo', size=(25, 1), text_color="red")]]


visualizacion = [[sg.TabGroup([[sg.Tab('Día 1', dia1, tooltip='tip'),
                sg.Tab('Día 2', dia2)]], tooltip='TIP2')]]

layout = [[sg.TabGroup([[sg.Tab('Plan', plan, tooltip='tip'),
                sg.Tab('Camiones', camiones, tooltip='TIP2'),
                sg.Tab('Demandas', demandas, tooltip='TIP3'),
                sg.Tab('Visualización', visualizacion)]], tooltip='TIP4')],#Tabs dentro de tab
            [sg.Button('Planificar'), sg.Button('Exit')]]

window = sg.Window('Planificador de rutas', layout, grab_anywhere=True)

while True:  # Event Loop
    event, values = window.read()
    #print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Planificar': 
        localidad = values[0]
        nCamiones = values[1]
        capCamiones = values[2]
        velCamiones = values[3]
        llenadoInicial = values[4]
        aumentoDiario = values[5]
        numDias = values[6]

        print(localidad, nCamiones, capCamiones,  velCamiones, llenadoInicial, aumentoDiario, numDias)    # the input data looks like a simple list when auto numbered

    
window.close()


