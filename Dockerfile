# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This is done first to leverage Docker's layer caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run server.py when the container launches
# We use 0.0.0.0 to make the server accessible from outside the container
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
