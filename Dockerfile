FROM python:3.9

# Set the working directory in the container
WORKDIR /hybrid_app

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all files from your local folder into the container (including CSV and code)
COPY . .

# Run the main python file
CMD ["streamlit", "run", "hybrid_app.py", "--server.port=8000", "--server.address=0.0.0.0"]

