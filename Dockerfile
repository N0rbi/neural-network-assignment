FROM gw000/keras-full:latest

COPY *.py .
COPY requirements.txt requirements.txt

ADD project/*.tar.gz project/

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
