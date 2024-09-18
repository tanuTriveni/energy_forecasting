# prediction_function.py

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

def predict_energy(state, start_date, end_date, model, df):
    if not model:
        return f"No model found for state: {state}", None, None, None

    date_range = pd.date_range(start=start_date, end=end_date)
    if date_range.empty:
        return "No data available for the specified date range.", None, None, None

    future = pd.DataFrame({'ds': date_range})
    forecast = model.predict(future)
    print(forecast)

    # Add actual data for validation if available
    test_df = future.merge(df[df['States'] == state][['Dates', 'Usage']], how='left', left_on='ds', right_on='Dates')
    actual_data = test_df['Usage'].dropna()
    forecast_data = forecast['yhat'][:len(actual_data)]

    if actual_data.empty:
        mse_value = "N/A (No actual data)"
        performance = "N/A"
    else:
        mse_value = mean_squared_error(actual_data, forecast_data)
        if mse_value > 100000:
            performance = "Poor"
        elif mse_value > 50000:
            performance = "Average"
        else:
            performance = "Best"

    result = pd.DataFrame({'Date': date_range, 'Predicted Usage': forecast['yhat']})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_range, y=forecast['yhat'], mode='lines', name='Predicted'))
    fig.update_layout(
        title='Energy Consumption Forecast',
        xaxis_title='Date',
        yaxis_title='Energy Consumption (kWh)',
        hovermode='x unified',
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["type", "scatter"],
                        label="Scatter Plot",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "bar"],
                        label="Bar Plot",
                        method="restyle"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.17,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    return result.to_string(index=False), f"MSE: {mse_value}", performance, fig
