# Import all modules
from flask import Flask, jsonify, request
from classifier import  get_prediction

# Defining the name (constructor)
app = Flask(__name__)

# Defining the method for the route
@app.route("/predict-alphabet", methods=["POST"])
# Predicting data ()
def predict_data():
  # image = cv2.imdecode(np.fromstring(request.files.get("digit").read(), np.uint8), cv2.IMREAD_UNCHANGED)
  image = request.files.get("alphabet")
  prediction = get_prediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

# Running (with the visibility of the new changes)
# "__main__" also prevents certain code being run when the model is imported
if __name__ == "__main__":
  app.run(debug=True)
