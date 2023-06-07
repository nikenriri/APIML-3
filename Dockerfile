FROM python:3.10.8

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#COPY mvp_model.h5 mvp_model.h5

COPY main.py main.py

ENV PYTHONUNBUFFERED=1 

ENV HOST 0.0.0.0

EXPOSE 8000

CMD [ "python", "main.py" ]
