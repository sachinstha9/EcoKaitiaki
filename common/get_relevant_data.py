from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import json

with open("assets/data.json", encoding='utf-8') as f:
    plants = json.load(f)

field_descriptions = {
    # "common name": "Name of the plant in everyday language",
    # "scientific name": "The scientific Latin name of the plant",
    # "description": "A short summary of the plant's key features",
    # "tags": "Keywords that describe the plant",
    "plant category": "General category like tree, shrub, groundcover, etc.",
    "plant type": "Is it perennial, native, or exotic",
    "mature height": "Maximum height the plant reaches when fully grown",
    "mature width": "Maximum width the plant spreads to",
    "growth rate": "How fast the plant grows",
    "soil type": "The kinds of soil suitable for this plant",
    "soil drainage": "How well the soil needs to drain for the plant",
    "soil ph": "The ideal pH levels of the soil for the plant",
    "location suitability": "What types of locations the plant grows well in",
    "climate zone": "Best suited climate zones (temperate, tropical, etc.)",
    "temperature range": "Range of temperatures the plant tolerates",
    "rainfall requirements": "How much rainfall the plant needs",
    "sun requirements": "Sunlight needs — full sun, partial shade, etc.",
    "salt tolerance": "How much salt exposure the plant can handle",
    "drought tolerance": "How well the plant handles drought",
    "frost tolerance": "How well the plant tolerates frost",
    "wind tolerance": "The level of wind the plant can endure",
    "flowering season": "The months when the plant produces flowers",
    "flower color": "Color of the plant's flowers",
    "maintenance level": "How easy or difficult the plant is to maintain",
    "wildlife value": "How the plant supports birds, insects, etc.",
    "uses": "Purposes such as erosion control, shade, decoration",
    "planting considerations": "Important things to know before planting",
    "companion plants": "Plants that grow well together with this one",
    "potential issues": "Possible downsides like toxicity or fragility",
    "places found": "Regions where this plant naturally grows",
    "best grown locations": "Cities or regions where this plant thrives best",
    "growing months": "The best months to grow or plant this species",
    "all": "all detailed informations about plants"
}

model = SentenceTransformer("all-MiniLM-L6-v2")

field_keys = list(field_descriptions.keys())
field_texts = list(field_descriptions.values())
field_embeddings = model.encode(field_texts, convert_to_tensor=True)


def get_relevant_fields(query, similarity_threshold = 0.6):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, field_embeddings)[0]
    relevant_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]
    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)

    ftrs = [field_keys[i] for i in sorted_indices]

    if "all" in ftrs:
        ftrs = field_keys
        ftrs.remove("all")

    return ["common name", "scientific name", "description", "tags"] + ftrs

def get_individual_text(features, data):
    text = ""
    for ftr in features:
        d = str(data.get(ftr, 'unknown')).replace("[", "").replace("]", "")
        if d != 'unknown': text += f"{ftr} is {d} . "
    return text

def get_relevant_data(query, max_k=10, score_threshold=0.45):  # cosine similarity threshold
    query = query.lower()
    features = get_relevant_fields(query)
    texts = [get_individual_text(features, plant) for plant in plants]

    vectors = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(len(vectors[0]))  # inner product = cosine sim if normalized
    index.add(np.array(vectors))

    query_vector = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vector), k=max_k)

    results = [texts[i] for i, score in zip(I[0], D[0]) if score > score_threshold]

    if not results:
        results = [texts[i] for i in I[0][:3]]

    return len(results), features, I, D

print(get_relevant_data("give me 3 plants that grow in yellow soil?"))