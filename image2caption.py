from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

MODEL1 = "Salesforce/blip-image-captioning-large"
MODEL2 = "nlpconnect/vit-gpt2-image-captioning"

@st.cache_resource(show_spinner=False)
def load_model1():
    # Loads blip-image-captioning-large model
    processor = BlipProcessor.from_pretrained(MODEL1)
    model = BlipForConditionalGeneration.from_pretrained(MODEL1)
    return model, processor

@st.cache_resource(show_spinner=False)
def load_model2():
    # Loads vit-gpt2-image-captioning model
    model2 = VisionEncoderDecoderModel.from_pretrained(MODEL2)
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL2)
    return model2, feature_extractor, tokenizer

def get_captions1(raw_img, model, processor):
    # Generates caption using BLIP model
    inputs = processor(raw_img, return_tensors="pt")
    gen_kwargs = {"max_new_tokens": 20}
    output = model.generate(**inputs, **gen_kwargs)
    pred = processor.decode(output[0], skip_special_tokens=True)
    return pred

def get_captions2(raw_img, model, feature_extractor, tokenizer):
    # Generates caption using GPT2 model
    pixel_values = feature_extractor(images=raw_img, return_tensors="pt").pixel_values
    gen_kwargs = {"max_new_tokens": 20}
    output = model.generate(pixel_values, **gen_kwargs)
    pred = tokenizer.batch_decode(output, skip_special_tokens=True)
    return pred[0]
   
if __name__== "__main__":
    with st.spinner('Loading Model'):
        blip_model, processor = load_model1()
        gpt2_model, feature_ext, tokenizer = load_model2()

    st.write("""
         # Image2Caption Generator
         """
         )
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])

    if(file != None):
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        with st.spinner('Generating Captions'):
            caption1 = get_captions1(image, blip_model, processor)
            caption2 = get_captions2(image, gpt2_model, feature_ext, tokenizer)
        st.success(caption1.capitalize())
        st.success(caption2.capitalize())
