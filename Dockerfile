FROM python:3.6.5-slim

LABEL maintainer="Christof Henkel <christof.henkel@khumbu.ai>"

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

# COPY <src> <dest>
COPY . ./toxic_comment_flask
WORKDIR /toxic_comment_flask
COPY ./app ./app
COPY ./app/LSTM_MultiAttention ./app/LSTM_MultiAttention
COPY ./src/models/LSTM_MultiAttention ./src/models/LSTM_MultiAttention


#RUN ls -la ./*
#RUN ls -la ./app/*
#RUN ls -la ./app/LSTM_MultiAttention/*
#RUN ls -la ./src/models/LSTM_MultiAttention/*


RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

EXPOSE 5001

CMD python -m app.app