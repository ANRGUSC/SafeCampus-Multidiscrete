import sys
import os
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add root folder to path
import dash_bootstrap_components as dbc
from dash import dcc, html
import dash
import plotly.graph_objs as go
import plotly.express as px
import time
from dash.dependencies import Input, Output, State
from main import run_training_and_evaluation, initialize_environment, load_config  # Import necessary functions


# Set shared config path and algorithm type
shared_config_path = os.path.join('config', 'config_shared.yaml')
csv_path = 'aggregated_weekly_risk_levels.csv'  # Set this to the actual CSV path

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
# Add dcc.Store to the layout for storing evaluation data
app.layout = html.Div([
    html.H1("Reinforcement Learning Dashboard", style={'textAlign': 'center'}),

    # Input box and progress bar in the middle
    html.Div([
        dcc.Input(id='alpha-input', type='number', placeholder='Enter alpha value', value=0.2,
                  style={'width': '200px'}),
        html.Button(id='submit-button', n_clicks=0, children='Run Evaluation', style={'margin-left': '10px'}),
        html.Div(id='progress-container', children=[
            dcc.Loading(id="loading-progress", children=[html.Div(id="progress-message")], type="default"),
            dbc.Progress(id="progress-bar", striped=True, animated=True, value=0,
                         style={'width': '400px', 'margin-top': '10px'}),
        ]),
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-bottom': '30px'}),

    # Store for evaluation data
    dcc.Store(id='eval-data-store'),

    # Two graphs side by side
    html.Div([
        dcc.Graph(id='evaluation-graph-infected-allowed', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='evaluation-graph-community-risk', style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    # Hidden interval component to update the graphs periodically
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0, disabled=True)
])

# Global variables to store data for the streaming plot
eval_data = {}
# Callback to handle training, showing progress, and plotting
@app.callback(
    [Output('progress-bar', 'value'),
     Output('progress-message', 'children'),
     Output('interval-component', 'disabled'),
     Output('eval-data-store', 'data')],
    [Input('submit-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('alpha-input', 'value')]
)
def update_output(n_clicks, n_intervals, alpha):
    # Initialize graphs as empty before training starts
    progress_value = 0
    progress_message = "Click 'Run' to start training."

    if n_clicks > 0:
        if n_intervals == 0:
            # Start training and evaluation at the beginning of the intervals (only once)
            try:
                # Initialize environment
                env, _ = initialize_environment(shared_config_path, algorithm='q_learning', mode='train')

                # Start the training and evaluation process once
                total_rewards, allowed_values_over_time, infected_values_over_time, community_risk_values = run_training_and_evaluation(
                    env, shared_config_path, alpha, 'q_learning', 'q_learning', csv_path)

                # Store evaluation results in a dictionary and return it to dcc.Store
                eval_data = {
                    'allowed_values': allowed_values_over_time,
                    'infected_values': infected_values_over_time,
                    'community_risk_values': community_risk_values
                }

                return progress_value, "Training in progress...", False, eval_data

            except Exception as e:
                return 0, f"An error occurred: {e}", True, None

        # Update progress bar incrementally
        progress_value = min(100, n_intervals * 10)  # Increment progress by 10 for each interval
        progress_message = f"Training in progress... {progress_value}%"

    return progress_value, progress_message, False, None

@app.callback(
    [Output('evaluation-graph-infected-allowed', 'figure'),
     Output('evaluation-graph-community-risk', 'figure')],
    [Input('eval-data-store', 'data')]
)
def update_graphs(eval_data):
    if eval_data is None:
        return {}, {}  # If no data, return empty plots

    # Retrieve data from the stored eval_data
    allowed_values_over_time = eval_data['allowed_values']
    infected_values_over_time = eval_data['infected_values']
    community_risk_values = eval_data['community_risk_values']

    # Create area charts for Infected and Allowed values over time
    infected_allowed_graph = {
        'data': [
            go.Scatter(x=list(range(len(infected_values_over_time))),
                       y=infected_values_over_time,
                       fill='tozeroy',
                       name="Infected Over Time"),
            go.Scatter(x=list(range(len(allowed_values_over_time))),
                       y=allowed_values_over_time,
                       fill='tonexty',
                       name="Allowed Over Time")
        ],
        'layout': go.Layout(title="Infected and Allowed Dynamics", xaxis_title="Time", yaxis_title="Value")
    }

    # Create line graph for Community Risk
    community_risk_graph = {
        'data': [
            go.Scatter(x=list(range(len(community_risk_values))),
                       y=community_risk_values,
                       mode='lines',
                       name="Community Risk Over Time")
        ],
        'layout': go.Layout(title="Community Risk Dynamics", xaxis_title="Time", yaxis_title="Risk Level")
    }

    return infected_allowed_graph, community_risk_graph


if __name__ == '__main__':
    # Run the app on port 8080
    app.run_server(debug=True, port=8080)
