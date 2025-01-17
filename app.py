# Libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template # helps create the web app
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder # to encode numerical labels
from sklearn.model_selection import train_test_split


# Initialize Flask app
application = Flask(__name__)


# Load the data and train the model
data = pd.read_csv("milknew.csv")
label_encoder = LabelEncoder()
data['Grade'] = label_encoder.fit_transform(data['Grade']) 
    # this is done as this column contains categorical data such as 'low', 'medium', etc.

# define features and target (Grade Column)
X = data.drop('Grade', axis=1)
y = data['Grade']

# split the data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating & training the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)



# Define the function (for predicting the Milk Quality)
def predicting_milkQuality(pH, Temprature, Taste, Odor, Fat, Turbidity, Colour):

    # this block of code is to convert strings into integers !
    Taste = int(Taste)
    Odor = int(Odor)
    Fat = int(Fat)
    Turbidity = int(Turbidity)

    # prepare the input for prediction
    features = np.array([[pH, Temprature, Taste, Odor, Fat, Turbidity, Colour]])
    prediction = knn.predict(features)

    predicted_label = label_encoder.inverse_transform(prediction) # convert number back to label
    return predicted_label[0] # returns the predicted label



# GUI Interface
# route to the home page (form)
@application.route("/")
def home():
    return render_template("index.html")  # render the form (frontend)


# route to the results page
@application.route("/predict", methods=["POST"])
def predict():

    # get data entered from the form
    pH = float(request.form['pH'])
    Temprature = float(request.form['Temprature'])
    Taste = int(request.form['Taste'])
    Odor = int(request.form['Odor'])
    Fat = int(request.form['Fat'])
    Turbidity = int(request.form['Turbidity'])
    Colour = float(request.form['Colour'])

    # make a prediction using the ML model
    prediction = predicting_milkQuality(pH, Temprature, Taste, Odor, Fat, Turbidity, Colour)

    # presents image based on the prediction
    if prediction == "high":
        image_url = "static/plus_plus.png"
    elif prediction == "medium":
        image_url = "static/plus.png"
    else:  # low quality
        image_url = "static/minus.png"

    # present the results page > template with the prediction label (capitalized) and designated image
    return render_template("result.html", prediction=prediction.capitalize(), image_url=image_url)



# Run Flask app
if __name__ == "__main__":
    application.run(debug=True) # starts the app only when app.py is run directly
