import torch
import torch.nn as nn
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import numpy as np

model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),

        nn.Linear(128, 64),
        nn.ReLU(),

        nn.Linear(64, 1),
        nn.Sigmoid()
)
model.load_state_dict(torch.load('models/model.pth'))
model.eval()


class Item(BaseModel):
    x: List[List[float]]

app = FastAPI()

@app.get("/")
def home():
    return {"message": "X and O Classifier is active"}

@app.post("/predict")
def predict(item: Item):
    X = np.array(item.x)

    X = torch.tensor(X, dtype=torch.float32)

    X = X.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(X)

    return {"prediction": output.item()}