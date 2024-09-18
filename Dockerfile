FROM python:3.9-slim

# set the working directory in the container
WORKDIR /CICD-on-GCP

#copy the contents into the container
COPY . /CICD-on-GCP

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#run the training script
CMD ["python", "train.py"]