from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

with open("assets/data.json", encoding='utf-8') as f:
    plants = json.load(f)


def get_relevant_data(query, k):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        f"{plant.get('common_name', 'Unknown plant')} ({plant.get('scientific_name', 'No scientific name')}) is a "
        f"{plant.get('growth_rate', 'unknown').lower()}-growing plant typically grown as "
        f"{', '.join(plant.get('plant_type', ['unknown'])).lower()} in the "
        f"{plant.get('plant_category', 'unknown').lower()} category. It grows to about "
        f"{plant.get('mature_height', 'unknown height')} tall and {plant.get('mature_width', 'unknown width')} wide. "
        f"It prefers {', '.join(plant.get('soil_drainage', ['unknown drainage']))} "
        f"{', '.join(plant.get('soil_type', ['unknown soil type'])).lower()} with a "
        f"{', '.join(plant.get('soil_ph', ['unknown pH'])).lower()} pH. Suitable for "
        f"{', '.join(plant.get('location_suitability', ['unknown locations'])).lower()} in "
        f"{', '.join(plant.get('climate_zone', ['unknown climate'])).lower()} climates. It thrives in "
        f"{plant.get('temperature_range', 'unknown temperature range')} with "
        f"{plant.get('rainfall_requirements', 'unknown rainfall').lower()} rainfall, and needs "
        f"{plant.get('sun_requirements', 'unknown sun requirements').lower()}. Salt tolerance is "
        f"{plant.get('salt_tolerance', 'unknown').lower()}, drought tolerance "
        f"{plant.get('drought_tolerance', 'unknown').lower()}, frost tolerance "
        f"{plant.get('frost_tolerance', 'unknown').lower()}, and wind tolerance "
        f"{plant.get('wind_tolerance', 'unknown').lower()}. Maintenance is "
        f"{plant.get('maintenance_level', 'unknown').lower()}. It can be used for "
        f"{', '.join(plant.get('uses', ['various uses'])).lower()}. Tips: "
        f"{plant.get('planting_considerations', 'No specific considerations.')}. Companion plants include "
        f"{', '.join(plant.get('companion_plants', ['none']))}. Possible issues: "
        f"{', '.join(plant.get('potential_issues', ['none']))}. Found in "
        f"{', '.join(plant.get('places_found', ['various locations']))}, best grown in "
        f"{', '.join(plant.get('best_grown_locations', ['various locations']))}. Growing months: "
        f"{', '.join(plant.get('growing_months', ['unknown']))}. Tags: "
        f"{', '.join(plant.get('tags', ['none']))}."
        for plant in plants
    ]

    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))

    query_vector = model.encode([query])

    D,I = index.search(np.array(query_vector), k = k)

    results = [plants[i] for i in I[0]]

    return results