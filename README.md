Plant Disease Detection using Deep Learning

Overview:

The project is a plant disease detection tool built using a neural network,and represented by a  simple Streamlit app so it's actually usable. 

How to use:
 upload a photo of a plant leaf, and the model tells you whether it looks healthy or shows signs of disease,and if so, which one.

 it's a Multi-Layer Perceptron (MLP) trained on a large public dataset of healthy and diseased leaves from a bunch of different crops. Once you upload an image, the app runs it through the model and gives you back a prediction along with a confidence score.

 Dataset:

We used the New Plant Diseases Dataset (Augmented) from Kaggle:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Info about the dataset:

- it has Roughly 87,000 RGB images in total
- contains 38 distinct classes (different crop/disease combinations, plus healthy examples)
which helps it Cover a wide range of crops apple, corn, grape, potato, tomato, peach, pepper, strawberry, soybean, cherry, orange, raspberry, squash, and blueberry
- Images are resized to 64×64 pixels before training the model 

How to run:
1. Clone the repo
git clone <repository-link>
cd Plant-Disease-Detection
2. Set up a virtual environment (recommended, but not required)
python -m venv venv
venv\Scripts\activate
3. Install the dependencies
pip install -r requirements.txt
4. Launch the app
streamlit run app.py


How the Model Works:

It's a simple MLP built with TensorFlow/Keras, trained on 64×64×3 images. Before an uploaded image reaches the model, it goes through a quick preprocessing step:


- Convert to RGB (in case it isn't already)
- Resize to 64×64
- Normalize pixel values (divide by 255)
- Add a batch dimension so it matches what the model expects



Output:

- The predicted disease (or "healthy," if that's the case)
- A confidence score for that prediction
- A few of the next most likely classes


Built With:

Python
TensorFlow / Keras
Streamlit
NumPy
Pillow (PIL)
Matplotlib
Plotly
Scikit-learn
