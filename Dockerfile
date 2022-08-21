FROM python:3.8-slim-buster
WORKDIR /app
# WORKDIR /artworks
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY classification.py classification.py
COPY artists.csv artists.csv
# COPY . .
CMD [ "python3", "classification.py"]
