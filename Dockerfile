FROM python:3.8-slim-buster
WORKDIR /app
# WORKDIR /artworks
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY prepare_csv.py prepare_csv.py
COPY resized/ resized/
# COPY . .
CMD [ "python3", "prepare_csv.py"]
