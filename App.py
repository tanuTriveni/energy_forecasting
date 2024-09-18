from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import pickle

app = Flask(__name__)


with open('state_models.pkl', 'rb') as f:
    state_models = pickle.load(f)

# Function to predict using the loaded model
def predict_energy(model, start_date, end_date):
    # Create a DataFrame for the future dates
    future_dates = pd.DataFrame({
        'ds': pd.date_range(start=start_date, end=end_date)
    })

    # Predict using the loaded model
    forecast = model.predict(future_dates)
    
    # Extract relevant information from the forecast
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    mse_value = 5.0  # Dummy MSE value for demonstration
    performance = "Good"  # Dummy performance for demonstration

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df['ds'], forecast_df['yhat'], marker='o')
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.2)
    plt.title(f'Predicted Energy Consumption from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Energy Usage')
    plt.grid(True)

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return forecast_df, mse_value, performance, image_base64

@app.route('/')
def home():
    # Render the form for user input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form['state']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Check if the state model exists
    if state not in state_models:
        return f"Model for state {state} not found.", 404

    model = state_models[state]

    # Perform prediction
    result, mse_value, performance, fig_base64 = predict_energy(model, start_date, end_date)

    return render_template('result.html', state=state, start_date=start_date, end_date=end_date,
                           result_table=result.to_html(classes='table table-striped'),
                           mse_value=mse_value, performance=performance, fig_base64=fig_base64)

if __name__ == '__main__':
    app.run(debug=True)
