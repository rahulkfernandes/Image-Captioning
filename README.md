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
