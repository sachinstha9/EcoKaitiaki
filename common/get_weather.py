import requests
import os

api_key = os.getenv("WEATHER_API_KEY")
if not api_key:
    raise ValueError("Please set the WEATHER_API_KEY environment variable")

def get_weather(lat, lon):
    location_query = f"{lat},{lon}"
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location_query}&days=3"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None
    
    data = response.json()

    total_precip = 0
    total_humidity = 0
    count = 0

    for day in data['forecast']['forecastday']:
        for hour in day['hour']:
            total_precip += hour.get('precip_mm', 0)
            total_humidity += hour.get('humidity', 0)
            count += 1

    if count == 0:
        print("No hourly data found.")
        return None

    avg_precip = total_precip / count
    avg_humidity = total_humidity / count

    if avg_precip > 4 and avg_humidity > 70:
        label = "Rainy and Humid"
    elif avg_precip < 1 and avg_humidity < 50:
        label = "Dry and Low Humidity"
    elif avg_precip < 1 and avg_humidity > 70:
        label = "Dry but Humid"
    elif avg_precip > 4 and avg_humidity < 50:
        label = "Wet but Low Humidity"
    elif 1 <= avg_precip <= 4:
        label = "Moderately Wet"
    else:
        label = "Moderate Conditions"

    result = {
        "average_precip_mm": round(avg_precip, 2),
        "average_humidity_percent": round(avg_humidity, 2),
        "weather_condition": label
    }

    return result

# if __name__ == "__main__":
#     
#     latitude = -36.853    
#     longitude = 174.769

#     weather_data = get_weather(latitude, longitude, api_key)
