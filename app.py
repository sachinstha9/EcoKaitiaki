from common.get_coords import get_coords
from common.get_weather import get_weather
from common.get_soil_type import get_soil_type

query = "Where is hamilton and auckland?"

coords = get_coords(query)

data = {}
for coord in coords:
    weather_data = get_weather(coord["lat"], coord["lng"])
    soil_data = get_soil_type(coord["lat"], coord["lng"])

    data[coord["input"]] = {
        "place": coord["resolved"],
        "lat": coord["lat"],
        "lng": coord["lng"],
        "average_precip_mm": weather_data["average_precip_mm"],
        "average_humidity_percent": weather_data["average_humidity_percent"],
        "weather_condition": weather_data["weather_condition"],
        "soil_type": soil_data
    }

print(data)