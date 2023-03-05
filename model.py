from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL = "Salesforce/blip-image-captioning-large"

def get_captions(img_path, model, processor):
    raw_image = Image.open(img_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    gen_kwargs = {"max_new_tokens": 20}
    out = model.generate(**inputs, **gen_kwargs)
    pred = processor.decode(out[0], skip_special_tokens=True)
    return pred

if __name__== "__main__":
    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(MODEL)
    caption = get_captions('Pictures/Image1.png', model, processor)
    print(caption.capitalize())
    caption = get_captions('Pictures/Image2.png', model, processor)
    print(caption.capitalize())
    caption = get_captions('Pictures/Image3.png', model, processor)
    print(caption.capitalize())