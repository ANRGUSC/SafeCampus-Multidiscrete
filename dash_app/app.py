import sys
import os
import numpy as np
import yaml
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# Import necessary functions from main.py
from main import run_training_and_evaluation, initialize_environment, load_config

# Set shared config path, algorithm type, and CSV path
shared_config_path = os.path.join(root_dir, 'config', 'config_shared.yaml')
csv_path = os.path.join(root_dir, 'aggregated_weekly_risk_levels.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("SafeCampus Dashboard", style={'textAlign': 'center'}),

    html.Div([
        dcc.Input(id='alpha-input', type='number', placeholder='Enter alpha value', value=0.2,
                  style={'width': '200px', 'marginRight': '10px'}),
        html.Button(id='submit-button', n_clicks=0, children='Run Simulation', style={'marginTop': '10px'}),
        html.Div(id='progress-container', children=[
            dcc.Loading(id="loading-progress", children=[html.Div(id="progress-message")], type="circle"),
        ], style={'marginTop': '20px'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginBottom': '30px'}),

    dcc.Store(id='simulation-data-store'),

    html.Div([
        html.Div([
            dcc.Graph(id='community-risk-graph', style={'height': '600px'}),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='infected-non-infected-graph', style={'height': '600px'}),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),

    dcc.Interval(id='interval-component', interval=300, n_intervals=0, disabled=True)
])

@app.callback(
    [Output('progress-message', 'children'),
     Output('interval-component', 'disabled'),
     Output('simulation-data-store', 'data'),
     Output('community-risk-graph', 'figure'),
     Output('infected-non-infected-graph', 'figure')],
    [Input('submit-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('alpha-input', 'value'),
     State('simulation-data-store', 'data')]
)
def update_simulation_and_graphs(n_clicks, n_intervals, alpha, simulation_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'submit-button' and n_clicks > 0:
        try:
            env, _ = initialize_environment(shared_config_path, algorithm='q_learning', mode='train')
            total_rewards, allowed_values, infected_values, community_risk_values = run_training_and_evaluation(
                env, shared_config_path, alpha, 'q_learning', 'q_learning', csv_path)

            simulation_data = {
                'total_rewards': total_rewards,
                'allowed_values': allowed_values,
                'infected_values': infected_values,
                'community_risk_values': community_risk_values,
                'current_index': 0
            }
            return ("Simulation complete. Streaming data...", False, simulation_data,
                    dash.no_update, dash.no_update)
        except Exception as e:
            return (f"An error occurred: {e}", True, None,
                    dash.no_update, dash.no_update)

    elif triggered_id == 'interval-component' and simulation_data is not None:
        current_index = simulation_data['current_index']
        if current_index >= len(simulation_data['community_risk_values']):
            return (dash.no_update, True, simulation_data,
                    dash.no_update, dash.no_update)

        # Community Risk Graph
        community_risk_fig = go.Figure()
        community_risk_fig.add_trace(go.Scatter(
            x=list(range(current_index + 1)),
            y=simulation_data['community_risk_values'][:current_index + 1],
            fill='tozeroy',
            name='Community Risk'
        ))
        community_risk_fig.update_layout(
            title='Community Risk Over Time',
            xaxis_title='Time',
            yaxis_title='Risk Level',
            height=600
        )

        # Infected/Non-Infected Graph
        infected_non_infected_fig = go.Figure()

        allowed = int(simulation_data['allowed_values'][current_index])
        infected = int(simulation_data['infected_values'][current_index])
        non_infected = allowed - infected

        # Generate positions for all 100 markers
        np.random.seed(42)  # Use a fixed seed for consistent positioning
        x_positions = np.random.rand(100)
        y_positions = np.random.rand(100)

        # Calculate the blinking effect
        blink_effect = 0.5 + 0.5 * np.sin(current_index * np.pi / 5)

        # Infected markers (red and blinking)
        infected_non_infected_fig.add_trace(go.Scatter(
            x=x_positions[:infected],
            y=y_positions[:infected],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='circle',
                opacity=blink_effect,
                line=dict(width=1, color='white')
            ),
            name='Infected'
        ))

        # Non-infected markers (green and blinking)
        infected_non_infected_fig.add_trace(go.Scatter(
            x=x_positions[infected:allowed],
            y=y_positions[infected:allowed],
            mode='markers',
            marker=dict(
                size=10,
                color='green',
                symbol='circle',
                opacity=blink_effect,
                line=dict(width=1, color='white')
            ),
            name='Non-Infected'
        ))

        # Inactive markers (gray and static)
        infected_non_infected_fig.add_trace(go.Scatter(
            x=x_positions[allowed:],
            y=y_positions[allowed:],
            mode='markers',
            marker=dict(
                size=10,
                color='gray',
                symbol='circle',
                opacity=0.5,
                line=dict(width=1, color='white')
            ),
            name='Inactive'
        ))

        # Update the layout
        infected_non_infected_fig.update_layout(
            title=dict(
                text=f'Current Status: {infected} Infected, {non_infected} Non-Infected, {100 - allowed} Inactive',
                font=dict(color='white')
            ),
            xaxis=dict(range=[-0.05, 1.05], showticklabels=False, title="", showgrid=False, zeroline=False),
            yaxis=dict(range=[-0.05, 1.05], showticklabels=False, title="", showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='white')
            ),
            height=500,
            width=500,
            plot_bgcolor='black',
            paper_bgcolor='black'
        )

        # Add a square shape to represent the room
        infected_non_infected_fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="white", width=2),
            fillcolor="rgba(0,0,0,0)",
        )

        simulation_data['current_index'] += 1

        return (dash.no_update, dash.no_update, simulation_data,
                community_risk_fig, infected_non_infected_fig)

    return ("Click 'Run Simulation' to start.", True, None,
            dash.no_update, dash.no_update)

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)