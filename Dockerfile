# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy requirements file first to leverage Docker caching
COPY requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire backend application into the container
COPY . /code

# Hugging Face Spaces run on port 7860 by default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]