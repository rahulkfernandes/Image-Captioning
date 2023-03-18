FROM python:3.10

EXPOSE 8080

WORKDIR /img2cap
COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "image2caption.py", "--server.port=8080", "--server.address=0.0.0.0"]