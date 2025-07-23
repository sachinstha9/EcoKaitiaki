import requests
from transformers import pipeline

from common.get_cords import get_cords

query = "What fruit trees can I plant in a sunny backyard with clay soil near Auckland?"
cords = get_cords(query)

auck_lat = cords[cords["city"] == "auckland"]["lat"]
auck_lng = cords[cords["city"] == "auckland"]["lng"]

def get_soil(lat, lon):
    url = f"https://rest.soilgrids.org/query?lon={lon}&lat={lat}&attributes=phh2o,clay"
    return requests.get(url).json().get("properties", {})
print(get_soil(auck_lat, auck_lng))


# candidate_labels = ["soil", "weather"]


# query_classification_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# result = query_classification_pipe(query, candidate_labels=candidate_labels)

