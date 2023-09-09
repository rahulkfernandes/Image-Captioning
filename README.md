# Image2Caption Generator

## Description
The Image2Caption Generator generates captions based on an uploaded image using the BLIP model (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation) and the Pegasus Paraphrase model.

## Installation
Clone Repository
```
git clone https://github.com/rahulkfernandes/Image-Captioning.git
```
Install Dependancies
```
pip install -r requirements.txt
```

## Usage
Run Streamlit Server
```
streamlit run image2caption.py
```
Build Docker Image
```
docker build -t [image name] .
```
Run Container
```
docker run -p [local port]:8080 [image name]
```
## Screenshots
<img width="1512" alt="Sample1" src="https://github.com/rahulkfernandes/Image-Captioning/assets/91873558/0d6e8529-98b0-45c0-8da3-65b82e20c727">
<img width="1512" alt="Sample2" src="https://github.com/rahulkfernandes/Image-Captioning/assets/91873558/09109860-0116-454f-b3e3-bfce2e5950bc">
