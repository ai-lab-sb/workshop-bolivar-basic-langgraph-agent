import json
import os
from google.oauth2.service_account import Credentials


def load_credentials() -> dict:
    """
    Reads a JSON file and returns its content as a Python dictionary.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "sb-iadaia-cap-dev-e91efbc5b66e.json")

    with open(file_path) as f:
        creds_dict = json.load(f)
    credentials = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return credentials



if __name__ == "__main__":
    creds = load_credentials()
    print(creds)
