import requests  # Third-party library
from globals import API_URL  # Global variable defined in another file
from utils import DataProcessor


def fetch_and_process_data():
    # Use the global variable for the request
    response = requests.get(API_URL)
    response.raise_for_status()

    raw_data = response.text

    # Use code from another file (utils.py)
    processor = DataProcessor()
    processed = processor.process_data(raw_data)
    processed = processor.add_prefix(processed)

    return processed


if __name__ == "__main__":
    result = fetch_and_process_data()
    print("Processed data:", result)
