import pandas as pd
import json

cords = pd.read_csv("assets/nz.csv")

with open('assets/soil.json', 'r') as file:
    data = json.load(file)

def get_soil_type(lat, lng):
    min_dist = []
    for i in range(len(cords.iloc[:, 0])):
        dist = pow(pow(lng - cords.iloc[i]["lng"], 2) + pow(lat - cords.iloc[i]["lat"], 2), 0.5)
        if i == 0:
            min_dist = [dist, cords.iloc[i]]
        if dist < min_dist[0]:
            min_dist = [dist, cords.iloc[i]]

    min_dist_city = min_dist[1]['city']

    return data[min_dist_city]