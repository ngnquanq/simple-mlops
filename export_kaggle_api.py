import json

def read_kaggle_json(file_path="kaggle.json"):
  """Reads the Kaggle JSON file and returns its contents as a dictionary.

  Args:
    file_path: The path to the Kaggle JSON file. Defaults to "kaggle.json".

  Returns:
    A dictionary containing the contents of the Kaggle JSON file.
  """

  try:
    with open(file_path, 'r') as f:
      data = json.load(f)
      return data
  except FileNotFoundError:
    print("Kaggle.json file not found.")
    return None

def export_to_cli(data):
  """Exports the specified information from the Kaggle JSON data to the CLI.

  Args:
    data: A dictionary containing the contents of the Kaggle JSON file.
  """

  if data:
    username = data.get('username', 'Unknown')
    api_token = data.get('key', 'Unknown')
    print(f"Username: {username}")
    print(f"API Token: {api_token}")

if __name__ == "__main__":
  kaggle_data = read_kaggle_json()
  export_to_cli(kaggle_data)