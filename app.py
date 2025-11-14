import os

# Redirect HuggingFace & Transformers cache
os.environ["HF_HOME"] = "E:/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "E:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/hf_cache"

# Redirect Streamlit cache
os.environ["STREAMLIT_CACHE_DIR"] = "E:/st_cache"

import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# -------------------------------------------------
# 1. DEVICE
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# 2. LOAD BLIP (Captioning)
# -------------------------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model

blip_processor, blip_model = load_blip()

def get_caption(img: Image.Image):
    inputs = blip_processor(images=img, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_new_tokens=64)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption


# -------------------------------------------------
# 3. LOAD SMALL LLM (GPT4-X-Alpaca, CPU-friendly)
# -------------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "chavinlo/gpt4-x-alpaca"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

llm = load_llm()


# -------------------------------------------------
# 4. LLM Q&A Function
# -------------------------------------------------
def ask_llm(caption, question):
    prompt = f"""
You are an AI assistant. You are given an image description and a question.
Answer concisely and accurately.

Image description: "{caption}"
Question: "{question}"
Answer:
    """

    output = llm(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    answer = output[0]["generated_text"].split("Answer:")[-1].strip()
    return answer


# -------------------------------------------------
# 5. STREAMLIT UI
# -------------------------------------------------
st.title("🧠 Vision + LLM Q&A Demo")
st.write("Upload an image → Ask a question → LLM answers based on the image.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
question = st.text_input("Your question about the image:")

if uploaded_file and question:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Generating caption..."):
        caption = get_caption(image)

    with st.spinner("Thinking..."):
        answer = ask_llm(caption, question)

    st.image(image, caption=caption, use_container_width=True)
    st.write("### 📝 Caption:", caption)
    st.write("### 💬 Answer:", answer)
