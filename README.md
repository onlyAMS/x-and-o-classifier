t4# X vs O Image Classifier

A simple deep learning project that classifies handwritten images as either X or O using a Convolutional Neural Network (CNN).
The model is trained on a subset of a larger alphabet dataset and served through a local FastAPI API.

### Dataset

The original dataset contains handwritten English alphabets.
Only the X and O classes were extracted for binary classification.

[A_Z dataset](https://www.kaggle.com/code/harshwardhanthakare/alphabet)

### Tech Stack

* Python
* PyTorch
* FastAPI
* NumPy
* Pandas
* Uvicorn

### Running the API

Start the FastAPI server:
**uvicorn api.main:app --reload**

Then open:
http://127.0.0.1:8000/docs

### Future Improvements

* Expand to full alphabet classification
* Improve accuracy with data augmentation
* Deploy API online
* Optimize model for faster inference