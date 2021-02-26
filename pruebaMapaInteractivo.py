import requests
import folium
import polyline

url = 'http://router.project-osrm.org/route/v1/driving/-2.61298409,43.15346938;-2.60903946,43.16835288?annotations=distance,duration'

def get_route(url):
    r = requests.get(url)
    res = r.json()

    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']
    
    out = {'route':routes,
           'start_point':start_point,
           'end_point':end_point,
           'distance':distance
          }

    return out

def get_map(route):
    
    m = folium.Map(location=[(route['start_point'][0] + route['end_point'][0])/2, 
                             (route['start_point'][1] + route['end_point'][1])/2], 
                   zoom_start=13)

    folium.PolyLine(
        route['route'],
        weight=8,
        color='blue',
        opacity=0.6
    ).add_to(m)

    folium.Marker(
        location=route['start_point'],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)

    folium.Marker(
        location=route['end_point'],
        icon=folium.Icon(icon='stop', color='red')
    ).add_to(m)

    return m

route = get_route(url)
mapa = get_map(route)
#https://www.thinkdatascience.com/post/2020-03-03-osrm/osrm/

#De momento solo crea un html en el directorio local.
mapa.save("prueba3.html")