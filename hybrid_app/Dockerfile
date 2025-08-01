FROM python:3.9-slim

WORKDIR /hybrid_app

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy data files
COPY collab_filtered_data.csv \
     interaction_matrix.npz \
     track_ids.npy \
     cleaned_data.csv \
     transformed_data.npz \
     transformed_hybrid_data.npz \
     data/

# Copy Python scripts
COPY hybrid_app.py \
     collaborative_filtering.py \
     content_based_filtering.py \
     hybrid_recommendations.py \
     data_cleaning.py \
     transform_filtered_data.py \
     ./

EXPOSE 8000

CMD ["streamlit", "run", "hybrid_app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
