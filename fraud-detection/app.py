from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the fraud detection model
model = LogisticRegression()
model.load('path/to/model/file.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        # Preprocess the data
        le = LabelEncoder()
        df['payment_method'] = le.fit_transform(df['payment_method'])
        df['country'] = le.fit_transform(df['country'])
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
        # Use the trained model to predict fraud
        y_pred = model.predict(X)
        df['fraud'] = y_pred
        return df.to_html()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
