from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer once, when the server starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

FEATURE_COLUMNS = [
    'job_title', 'company_name', 'company_desc', 'job_desc',
    'job_requirement', 'salary', 'location', 'employment_type', 'department'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Build the same combined text format used during training
    combined_text = ' '.join(
        str(data.get(col, '') or '') for col in FEATURE_COLUMNS
    )

    # Convert to the same numeric format the model expects
    text_vector = vectorizer.transform([combined_text])

    # predict_proba returns [probability_of_class_0, probability_of_class_1]
    # class 1 = fake job, so we want index 1
    fraud_probability = model.predict_proba(text_vector)[0][1]

    return jsonify({
        "fraud_probability": round(float(fraud_probability), 4)
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)