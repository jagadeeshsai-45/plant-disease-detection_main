import requests

url = "http://127.0.0.1:5000/predict"
file_path = "D:\\PlantVillage\\Tomato_healthy\\0a205a11-1e64-49f7-93c2-ad59312b4f83___RS_HL 0334.JPG"
with open(file_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.json())  # Print API response
