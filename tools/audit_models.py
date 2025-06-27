import os, json, re

BASE      = os.path.dirname(__file__)
MODELS    = os.path.join(BASE, "..", "Models")
PRODUCTS  = os.path.join(BASE, "..", "data", "product_list.json")

def clean(name):
    name = re.sub(r'[^\w\s]', '', name)
    return re.sub(r'\s+', '_', name)

with open(PRODUCTS, encoding="utf-8") as f:
    product_map = json.load(f)

missing = []

for category, plist in product_map.items():
    for p in plist:
        pc = clean(p)
        files = {
            "arima" : f"{pc}_arima_model.pkl",
            "rf"    : f"{pc}_rf_model.pkl",
            "lstm"  : f"{pc}_lstm_model.h5",
            "scaler": f"{pc}_scaler.pkl"
        }
        absent = [k for k,v in files.items() if not os.path.exists(os.path.join(MODELS, v))]
        if absent:
            missing.append((p, absent))

print("=== Missing model assets ===")
for prod, absent in missing:
    print(f"{prod}: {', '.join(absent)}")
