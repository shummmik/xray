FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /app
WORKDIR /app
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY ./main.py ./
COPY ./model.py ./
COPY ./model.pt ./

EXPOSE 8501
CMD streamlit run main.py