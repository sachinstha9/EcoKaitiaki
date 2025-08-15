from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from sentence_transformers import (
    SentenceTransformer, 
    util
)
from common.GetSoilType import get_soil_type
from common.GetLeafDisease import get_leaf_disease
from common.GetRelevantData import get_relevant_data
import torch
import threading
import gradio as gr
from PIL import Image

# ----------------- Model setup -----------------
model_nm = "mistralai/Mistral-7B-Instruct-v0.3"
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
# Some causal LMs don't define a pad token. Set it explicitly to avoid warnings.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_nm,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)


def is_query_related_embedding(user_query, history, threshold=0.7):
    if not history:
        return False

    query_emb = embed_model.encode(user_query, convert_to_tensor=True, device="cuda")

    for user_msg, bot_msg in history:
        history_text = user_msg + " " + bot_msg
        history_emb = embed_model.encode(history_text, convert_to_tensor=True)
        similarity = util.cos_sim(query_emb, history_emb).item()
        if similarity >= threshold:
            return True

    return False


def build_prompt(history, user_query, plant_results):
    history = history[-10:] if history else []

    context = ""
    for plant in plant_results:
        context += plant + "\n\n"

    chat_history_str = ""
    if is_query_related_embedding(user_query, history):
        for user_msg, bot_msg in history[-5:]:
            chat_history_str += f"User: {user_msg}\nAnswer: {bot_msg}\n"

    prompt = f"""
You are a friendly and knowledgeable botanist specializing in New Zealand plants.

Use ONLY the information in the context to answer the user’s question. Do not make up facts.

Instructions for your answer:
- Start your answer with a natural, friendly acknowledgment, but avoid repeating the user's question verbatim.
- Keep it concise and flowing—don’t begin with “Hello there” or restate location unless necessary.
- Present plant suggestions in a readable, conversational style.
- For each plant, include:
    - Name (and scientific name if available)
    - Type (vegetable, tree, etc.)
    - Key growth details (height, spread, any special notes)
    - Ideal soil type, pH, and important care tips
- Avoid repetitive phrases like "with a growth rate" or "provided in the context."
- End with a friendly conclusion, e.g., “Happy gardening!” or “These plants should thrive in your garden.”
- If context is empty, simply say “I don’t know.”

Conversation so far:
{chat_history_str}

User’s question:
{user_query}

Context:
{context}
"""
    return prompt

def generate_streaming(history, user_query):
    try:
        plants_found = get_relevant_data(user_query)
    except Exception as e:
        plants_found = []
        print(f"[warn] get_relevant_data failed: {e}")

    prompt = build_prompt(history, user_query, plants_found)
    print(prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_output = ""
    for new_text in streamer:
        partial_output += new_text
        yield partial_output  # stream incrementally to the UI

    thread.join()

def chat_fn(message, history, uploaded_img=None, img_type=None):
    history = history or []

    image_info = ""
    if uploaded_img and img_type:
        img_path = ""
        uploaded_img.save(img_path)
        try:
            if img_type == "soil":
                image_info = f"The soil type is {get_soil_type(img_path)}. "
            elif img_type == "leaf":
                image_info = f"The leaf disease is {get_leaf_disease(img_path)}. "
        except Exception as e:
            print(f"[warn] image processing failed: {e}")

    final_query = (image_info + message) if image_info else message

    history.append([message, ""])
    bot_index = len(history) - 1

    partial_output = ""
    for new_text in generate_streaming([], final_query):  
        partial_output += new_text
        history[bot_index][1] = partial_output
        yield history 

with gr.Blocks() as demo:
    gr.Markdown("🌱 **EcoKaitiaki Plant Recommendation Chatbot**")
    
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your question here...")
        img_upload = gr.Image(type="pil", label="Upload Soil/Leaf Image (Optional)")
        img_type = gr.Radio(choices=["soil", "leaf"], label="Select image type (if uploaded)", value=None)
        submit_btn = gr.Button("Send")

    def submit_fn(message, history, uploaded_img, img_type):
        print(history)
        # return chat_fn(message, history, uploaded_img, img_type)
        return True
    
    submit_btn.click(
        fn=submit_fn,
        inputs=[msg, chatbot, img_upload, img_type],
        outputs=[chatbot],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch()
