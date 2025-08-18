import requests, base64, json

API = "http://127.0.0.1:8000"

# send to /edges and save the PNG
with open("sample_input.png", "rb") as f:
    files = {"file": ("sample_input.png", f, "image/png")}
    r = requests.post(f"{API}/edges?low=50&high=150", files=files)
    r.raise_for_status()
    with open("edges_out.png", "wb") as out:
        out.write(r.content)
print("Saved edges_out.png")

# send to /edges/json and save PNG from base64
with open("sample_input.png", "rb") as f:
    files = {"file": ("sample_input.png", f, "image/png")}
    r = requests.post(f"{API}/edges/json?low=80&high=200", files=files)
    r.raise_for_status()
    data = r.json()
    print("JSON keys:", list(data.keys()))
    png_b64 = data["image_png_base64"]
    with open("edges_from_json.png", "wb") as out:
        out.write(base64.b64decode(png_b64))
print("Saved edges_from_json.png")
