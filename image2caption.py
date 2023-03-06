import re
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

MODEL1 = "Salesforce/blip-image-captioning-large"
MODEL2 = "tuner007/pegasus_paraphrase"

@st.cache_resource(show_spinner=False)
def load_models():
    # Loads blip-image-captioning-large model & tuner007/pegasus_paraphrase
    processor = BlipProcessor.from_pretrained(MODEL1)
    blip_model = BlipForConditionalGeneration.from_pretrained(MODEL1)

    tokenizer = PegasusTokenizer.from_pretrained(MODEL2)
    para_model = PegasusForConditionalGeneration.from_pretrained(MODEL2)
    return blip_model, processor, para_model, tokenizer

def get_captions(raw_img, model, processor):
    # Generates caption using BLIP model
    inputs = processor(raw_img, return_tensors="pt")
    gen_kwargs = {"max_new_tokens": 20}
    output = model.generate(**inputs, **gen_kwargs)
    pred = processor.decode(output[0], skip_special_tokens=True)
    return pred

def get_response(input_text, model, tokenizer, num_return_sequences, num_beams):
    # Generates Paraphrases
    batch = tokenizer(
        [input_text],
        truncation=True,
        padding='longest',
        max_length=60,
        return_tensors="pt"
    )
    translated = model.generate(
        **batch,
        max_length=60,
        num_beams=num_beams, 
        num_return_sequences=num_return_sequences, 
        temperature=1.5
    )
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text
   
if __name__== "__main__":
    with st.spinner('Loading'):
        blip_model, processor, para_model, tokenizer = load_models()

    st.write("""
         # Image2Caption Generator
         """
         )
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])

    if(file != None):
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        with st.spinner('Generating Captions'):
            caption = get_captions(image, blip_model, processor).capitalize()
            num_beams = 10
            num_return_sequences = 4
            paraphrases = get_response(
                caption,
                para_model, 
                tokenizer, 
                num_return_sequences,
                num_beams
            )
        paraphrases = str(paraphrases)
        paraphrases = re.sub(r'[.\[\]\']', '', paraphrases)
        all_phrases = caption + ', ' + paraphrases
        st.success(all_phrases)