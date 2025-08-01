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
    context =  "" 
    for plant in plant_results:
        context += plant + "\n\n"
    
    prompt = (
        f"You are a knowledgeable plant expert AI. Based on the user query, "
        f"recommend the most suitable plants from the options below.\n\n"
        f"User Query: {query}\n\n"
        f"Plant Options:\n{context}\n\n"
        f"Please recommend the best plant(s) and explain your choice."
    )
    return prompt

def rag_plant_advisor(query):
    plants_found = get_relevant_data(query)
    prompt = build_prompt(query, plants_found)
    # response = generator(prompt)[0]["generated_text"]
    # answer = response[len(prompt):].strip()
    return prompt

output = rag_plant_advisor("give me plants that grows in rainy seasons.")
print(output)
