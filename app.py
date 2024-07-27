from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def load_and_train_model():
    data = pd.read_csv('synthetic_sensor_data.csv')
    X = data.drop('failure', axis=1)
    y = data['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

model, model_accuracy = load_and_train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/launch', methods=['POST'])
def launch():
    data = request.json
    features = pd.DataFrame([data])
    prediction = model.predict(features)[0]
    response = {
        "status": "Rocket launched!" if prediction == 1 else "Launch failed!",
        "accuracy": model_accuracy
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
