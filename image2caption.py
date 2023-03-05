from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL = "Salesforce/blip-image-captioning-large"

def get_captions(raw_img, model, processor):
    inputs = processor(raw_img, return_tensors="pt")
    gen_kwargs = {"max_new_tokens": 20}
    out = model.generate(**inputs, **gen_kwargs)
    pred = processor.decode(out[0], skip_special_tokens=True)
    return pred

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(MODEL)
    return model, processor

if __name__== "__main__":
    with st.spinner('Loading Model'):
        model, processor=load_model()

    st.write("""
         # Image2Caption Generator
         """
         )
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])

    if(file != None):
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        with st.spinner('Generating Caption'):
            caption = get_captions(image, model, processor).capitalize()
        st.success(caption)