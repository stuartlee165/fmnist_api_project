import requests
from requests.auth import HTTPBasicAuth

url = "http://localhost:5000/upload"
file_path = "test_1_image.png"

with open(file_path, "rb") as img:
    name_img = file_path.split("/")[-1]
    files = {"image": (name_img, img, "multipart/form-data", {"Expires": "0"})}
    with requests.Session() as s:
        r = s.post(url, files=files, auth=HTTPBasicAuth("user1", "password1"))
    print(r.status_code, r.text)
