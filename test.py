import requests

url = "http://localhost:5000/analyze"

with open("ramu.jpg", "rb") as f:
    response = requests.post(url, 
        files={"file": f},
        data={"name": "TestUser", "reason": "testing"}
    )

print(response.json())