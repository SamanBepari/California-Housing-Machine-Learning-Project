from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("Linear_Regression_California.pkl")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>California Housing Price Prediction</title>
    <style>
        * {
            box-sizing: border-box;
        }

        

        body {
        margin: 0;
        font-family: "Segoe UI", Tahoma, sans-serif;
        background: linear-gradient(135deg, #4f46e5, #9333ea);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        }

        .container {
            width: 480px;
            padding: 35px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            color: #fff;
        }

        h2 {
            text-align: center;
            margin-bottom: 25px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        label {
            font-size: 13px;
            opacity: 0.9;
        }

        input {
            width: 100%;
            padding: 10px 12px;
            border-radius: 8px;
            border: none;
            outline: none;
            font-size: 14px;
        }

        input::placeholder {
            color: #aaa;
        }

        input:focus {
            box-shadow: 0 0 0 2px #c7d2fe;
        }

        .full {
            grid-column: span 2;
        }

        button {
            margin-top: 10px;
            grid-column: span 2;
            padding: 14px;
            background: linear-gradient(135deg, #22c55e, #16a34a);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(34, 197, 94, 0.4);
        }

        .result {
            margin-top: 25px;
            padding: 18px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            text-align: center;
        }

        .result h3 {
            margin: 0;
            color: #dcfce7;
            font-weight: 600;
        }

        footer {
            text-align: center;
            margin-top: 18px;
            font-size: 12px;
            opacity: 0.8;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>🏡 California House Price Predictor</h2>

    <form method="post">
        <div>
            <label>Median Income</label>
            <input type="number" step="any" name="MedInc" placeholder="e.g. 4.5" required>
        </div>

        <div>
            <label>House Age</label>
            <input type="number" step="any" name="HouseAge" placeholder="e.g. 20" required>
        </div>

        <div>
            <label>Average Rooms</label>
            <input type="number" step="any" name="AveRooms" placeholder="e.g. 6.2" required>
        </div>

        <div>
            <label>Average Bedrooms</label>
            <input type="number" step="any" name="AveBedrms" placeholder="e.g. 1.1" required>
        </div>

        <div>
            <label>Population</label>
            <input type="number" step="any" name="Population" placeholder="e.g. 1500" required>
        </div>

        <div>
            <label>Avg Occupancy</label>
            <input type="number" step="any" name="AveOccup" placeholder="e.g. 3.0" required>
        </div>

        <button type="submit">Predict Price</button>
    </form>

    {% if prediction %}
    <div class="result">
        <h3>Estimated Value: $ {{ prediction }}</h3>
    </div>
    {% endif %}

    <footer>
        Linear Regression · California Housing Dataset
    </footer>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
        ]

        input_data = np.array(features).reshape(1, -1)
        prediction = round(model.predict(input_data)[0], 2)

    return render_template_string(HTML_PAGE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
