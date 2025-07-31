# set up the base image
FROM python:3.11

# set the working directory
WORKDIR /app/

# copy the requirements file to workdir
COPY requirements.txt .

# install the requirements
RUN pip install -r requirements.txt

# Copy all required data files at once
COPY ./collab_filtered_data.csv \
     ./interaction_matrix.npz \
     ./track_ids.npy \
     ./cleaned_data.csv \
     ./transformed_data.npz \
     ./transformed_hybrid_data.npz \
     ./data/


# Copy all required Python scripts at once
COPY hybrid_app.py \
     collaborative_filtering.py \
     content_based_filtering.py \
     hybrid_recommendations.py \
     data_cleaning.py \
     transform_filtered_data.py \
     ./

# expose the port on the container
EXPOSE 8000

# run the streamlit app
CMD [ "streamlit", "run", "hybrid_app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]