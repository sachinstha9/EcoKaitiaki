from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from common.GetSoilType import get_soil_type
from common.GetLeafDisease import get_leaf_disease
from common.GetRelevantData import get_relevant_data
import torch

query = ""
soil_img_path = "assets\download (5).jfif"
leaf_img_path = ""
img_class = "soil"

if soil_img_path != "":
    soil_type = get_soil_type(img_path=soil_img_path)
    query += f"The soil type is {soil_type} . "

if leaf_img_path != "":
    leaf_disease = get_leaf_disease(img_path=leaf_img_path)
    query += f"The leaf disease is {leaf_disease} . "

model_nm = "google/gemma-7b-it"

def build_prompt(query, plant_results):
    context =  "" 
    for plant in plant_results:
        context += plant + "\n\n"
    
    prompt = (
        f"You are a knowledgeable plant expert. "
        f"Recommend the most suitable plants from the options below.\n\n"
        f"User Query: {query}\n\n"
        f"Plant Options:\n{context}\n\n"
        f"If the option doesnot match with the query then say just I donot know."
        f"Please recommend the best plant(s) and explain your choice."
        f"Answer the question as a friendly teacher."
    )
    return prompt

tokenizer = AutoTokenizer.from_pretrained(model_nm)
model = AutoModelForCausalLM.from_pretrained(
    model_nm,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7
)

def get_response(query):
    plants_found = get_relevant_data(query)
    prompt = build_prompt(query, plants_found)
    response = generator(prompt)[0]["generated_text"]
    answer = response[len(prompt):].strip()
    return answer

print(get_response("i live in auckland, so which are the best plants i could plant in my home. I am a very busy person, so i donot have much time to care for the plants, so suggest me some plants."))