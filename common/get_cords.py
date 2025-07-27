from transformers import pipeline
import requests
import os

api_key = os.getenv("OPENCAGE_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENCAGE_API_KEY environment variable")

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_locations(query):
    entities = ner_pipeline(query)
    locations = [ent['word'] for ent in entities if ent['entity_group'] in ['LOC', 'ORG', 'PER', 'MISC']]
    return list(set(locations)) 

def geocode_location(location):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        "q": location,
        "key": api_key,
        "countrycode": "nz",
        "limit": 1,
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["results"]:
        result = data["results"][0]
        return {
            "input": location,
            "resolved": result["formatted"],
            "lat": result["geometry"]["lat"],
            "lng": result["geometry"]["lng"]
        }
    return {"input": location, "error": "Not found"}

def process_query(query):
    locations = extract_locations(query)
    print(f"📍 Found locations: {locations}")
    results = [geocode_location(loc) for loc in locations]
    return results

query = "How do I get from Symnds St to Queen Street in Auckland?"
output = process_query(query)

for loc in output:
    print(loc)
