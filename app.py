from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Initialize Flask app
application = Flask(__name__)


# Load the data and train the model
data = pd.read_csv("milknew.csv")
label_encoder = LabelEncoder()
data['Grade'] = label_encoder.fit_transform(data['Grade'])

X = data.drop('Grade', axis=1)
y = data['Grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# Define the function (for predicting the Milk Quality)
def predicting_milkQuality(pH, Temprature, Taste, Odor, Fat, Turbidity, Colour):

    # This block of code is to convert gr.radio strings into integers !
    Taste = int(Taste)
    Odor = int(Odor)
    Fat = int(Fat)
    Turbidity = int(Turbidity)

    # Prepare the input for prediction
    features = np.array([[pH, Temprature, Taste, Odor, Fat, Turbidity, Colour]])
    prediction = knn.predict(features)

    predicted_label = label_encoder.inverse_transform(prediction) # Convert number back to label
    return predicted_label[0]



# GUI Interface
# Route for the home page (form)
@application.route("/")
def home():
    return render_template("index.html")  # Render the form (frontend)


# Route for the results page
@application.route("/predict", methods=["POST"])
def predict():

    # Get data from the form
    pH = float(request.form['pH'])
    Temprature = float(request.form['Temprature'])
    Taste = int(request.form['Taste'])
    Odor = int(request.form['Odor'])
    Fat = int(request.form['Fat'])
    Turbidity = int(request.form['Turbidity'])
    Colour = float(request.form['Colour'])

    # Make a prediction using the ML model
    prediction = predicting_milkQuality(pH, Temprature, Taste, Odor, Fat, Turbidity, Colour)


    # result image based on the prediction
    if prediction == "high":
        image_url = "static/plus_plus.png"
    elif prediction == "medium":
        image_url = "static/plus.png"
    else:  # low quality
        image_url = "static/minus.png"

    # present the results page
    return render_template("result.html", prediction=prediction.capitalize(), image_url=image_url)


# Run Flask app
if __name__ == "__main__":
    application.run(debug=True)
