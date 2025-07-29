from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

with open("assets/data.json", encoding='utf-8') as f:
    plants = json.load(f)


def get_relevant_data(query, k):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        f'{p.get("common_name", "Unknown")} ({p.get("scientific_name", "N/A")}): '
        f'{p.get("description", "")}. Grows in {p.get("soil_type", "varied soils")} '
        f'with {p.get("drought_tolerance", "moderate")} drought tolerance.'
        for p in plants
    ]

    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))

    query_vector = model.encode([query])

    D,I = index.search(np.array(query_vector), k = k)

    results = [plants[i] for i in I[0]]

    return results