# artworks_classifier

Machine learning project to classify artworks

### To run container

docker build --tag python-docker .
docker images
docker run -v "$(pwd)/csv_data:/app/csv_data" python-docker

### To prepare data

1. python3 prepare_data.py
2. preparocess_data.py
