from common.get_relevant_data import get_relevant_data
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    torch_dtype=torch.float16,
    trust_remote_code=True  
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2000,
    do_sample=True,
    temperature=0.7,
)

def build_prompt(query, plant_results):
    context =  [
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
            f"{', '.join(plant.get('tags', ['none']))}.\n\n"
            for plant in plant_results
        ]
    
    prompt = (
        f"You are a knowledgeable plant expert AI. Based on the user query, "
        f"recommend the most suitable plants from the options below.\n\n"
        f"User Query: {query}\n\n"
        f"Plant Options:\n{context}\n\n"
        f"Please recommend the best plant(s) and explain your choice."
    )
    return prompt

def rag_plant_advisor(query):
    plants_found = get_relevant_data(query, 3)

    prompt = build_prompt(query, plants_found)

    response = generator(prompt)[0]["generated_text"]

    answer = response[len(prompt):].strip()

    return answer

import gradio as gr

demo = gr.Interface(
    rag_plant_advisor,
    inputs=["text"],
    outputs=["text"]
)

demo.launch()
