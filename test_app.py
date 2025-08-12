import requests
import time

app_url = "http://localhost:8501"

def get_app_status(url):
    response = requests.get(url)
    status_code = response.status_code
    return status_code

def test_app_loading():
    time.sleep(10) # Give the app time to load
    status_code = get_app_status(app_url)
    assert status_code == 200, "Unable to load Streamlit App"
    print("Streamlit App Loaded Successfully")