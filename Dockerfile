FROM python
COPY . /usr/src/app
WORKDIR /usr/src/app/python
RUN apt-get update && apt-get install libgl1-mesa-dev --yes
RUN pip install -r requirements.txt