FROM python:3.6.5-slim

LABEL maintainer="Christof Henkel <christof.henkel@khumbu.ai>"

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /toxic_comment_flask
WORKDIR /toxic_comment_flask
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

EXPOSE 5001
#CMD flask run --host=0.0.0.0
ENTRYPOINT ["python"]
CMD ["main.py"]