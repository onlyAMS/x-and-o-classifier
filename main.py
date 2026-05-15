import requests
import pandas as pd

data = pd.read_csv('data/val_data.csv')


X = data.drop(columns=['y']).values / 225
X = X.reshape(-1, 1, 28, 28)
y = data['y'].values

url = 'http://127.0.0.1:8000/predict'

idx =78
data = {
    'x': X[idx][0].tolist()
}

response = requests.post(url, json=data)
output = response.json()
prediction = output['prediction']
print(f"prediction: {prediction}")

if prediction >= 0.5:
    print("Image is an X")
else:
    print('Image is an O')

test = y[idx]
if test == 0:
    print('Image is actually an O')
else:
    print('Image is an actually an X')

