"""
Dash app to configure and run the supply-chain simulation using a JSON config textarea.

Usage: pip install dash pandas openpyxl
Run: python app.py
"""
import json
import io
import traceback
import os
from datetime import datetime

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

# Import simulation building blocks
from main import get_results
from Simulation import Simulation
from Consumer import Consumer
from Transhipper import Transhipper
from Producer import Producer
from op_base_stock_fr_all_first import op_base_stock_fr_all_first
from pp_base_stock import pp_base_stock
from ap_proportional import ap_proportional

# --- Helpers ---
DEFAULT_CONFIG = {
    "consumers": [
        {"name": "H1", "d": 150, "dstd": 10, "ss": 1000, "suppliers": [0]},
        {"name": "H2", "d": 170, "dstd": 10, "ss": 1000, "suppliers": [0]},
    ],
    "transhippers": [
        {"name": "WS1", "suppliers": [0], "customers": [0, 1], "ss": 8000, "l": 2},
    ],
    "producers": [
        {"name": "MN1", "ss": 10000, "m": 800, "l": 2, "pl": 2, "customers": [0]},
    ],
}
DEFAULT_CONFIG_TEXT = json.dumps(DEFAULT_CONFIG, indent=2)


def build_simulation_from_config(sim_periods, config_json):
    """Build Simulation object from config JSON string.
    If config_json is empty or invalid, uses DEFAULT_CONFIG.
    """
    try:
        if not config_json or config_json.strip() == "":
            config = DEFAULT_CONFIG
        else:
            config = json.loads(config_json)
    except Exception:
        config = DEFAULT_CONFIG

    # Build consumers
    consumers = []
    for c in config.get("consumers", []):
        consumers.append(
            Consumer(
                name=c.get("name"),
                d=c.get("d", 0),
                dstd=c.get("dstd", 0),
                ss=c.get("ss", 0),
                suppliers=c.get("suppliers", []),
                order_policy_function=op_base_stock_fr_all_first,
            )
        )

    # Build transhippers
    transhippers = []
    for t in config.get("transhippers", []):
        transhippers.append(
            Transhipper(
                consumers=consumers,
                name=t.get("name"),
                suppliers=t.get("suppliers", []),
                customers=t.get("customers", []),
                ss=t.get("ss", 0),
                l=t.get("l", 1),
                order_policy_function=op_base_stock_fr_all_first,
                allocation_policy_function=ap_proportional,
            )
        )

    # Build producers
    producers = []
    for p in config.get("producers", []):
        producers.append(
            Producer(
                transhippers=transhippers,
                name=p.get("name"),
                ss=p.get("ss", 0),
                m=p.get("m", 0),
                l=p.get("l", 1),
                pl=p.get("pl", 1),
                customers=p.get("customers", []),
                production_policy_function=pp_base_stock,
                allocation_policy_function=ap_proportional,
            )
        )

    def disruption_function(self, t):
        pass

    def change_decision_policies(self, t):
        pass

    simulation = Simulation(sim_periods, consumers, transhippers, producers, disruption_function, change_decision_policies)
    return simulation


# --- Dash App --#
app = Dash(__name__)
server = app.server

last_updated_path = os.path.join(os.path.dirname(__file__), "main.py")
try:
    last_updated_ts = os.path.getmtime(last_updated_path)
    last_updated = datetime.fromtimestamp(last_updated_ts).strftime("%Y-%m-%d %H:%M:%S")
except Exception:
    last_updated = "Unknown"

CREATOR = "Noah Chicoine (ncc1203)"


app.layout = html.Div(className="app-container", children=[
    # Title section
    html.Div(className="title-section", children=[
        html.H1("Supply Chain Simulation Runner"),
        html.Div(f"Creator: {CREATOR}"),
        html.Div(f"Last updated: {last_updated}", className="meta"),
    ]),

    # Simulation setup
    html.Div(className="setup-section", children=[
        html.H3("Simulation Setup"),
        html.Div(className="controls", children=[
            html.Div([html.Label("Simulation periods:"), dcc.Input(id="sim-periods", type="number", value=300, min=1, step=1)], className="control-item"),
            html.Div([html.Button("Use Defaults", id="use-defaults", n_clicks=0)], className="control-item"),
        ], style={"display": "flex", "gap": "12px", "alignItems": "center"}),

        html.Div([
            html.Label("Simulation config (JSON)"),
            dcc.Textarea(id="config-json", value=DEFAULT_CONFIG_TEXT, style={"width": "100%", "height": "260px"}),
            html.Div(style={"marginTop": "8px"}, children=[
                html.Button("Run Simulation", id="run-button", n_clicks=0, className="run-button"),
            ]),
        ]),
    ]),

    # Simulation results visuals
    html.Div(className="results-section", children=[
        html.H3("Simulation Results"),
        html.Div(id="run-status", className="run-status"),
        dcc.Graph(id="inventory-graph", className="inventory-graph"),
        html.Div([html.Button("Download Results (Excel)", id="download-button", n_clicks=0, className="download-button"), dcc.Download(id="download")], style={"marginTop": "12px"}),
    ]),

    # Store results as JSON
    dcc.Store(id="results-store"),
])


@app.callback(Output("config-json", "value"), Input("use-defaults", "n_clicks"))
def fill_defaults(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return DEFAULT_CONFIG_TEXT


@app.callback(Output("run-status", "children"),
                Output("results-store", "data"),
                Output("inventory-graph", "figure"),
                Input("run-button", "n_clicks"),
                State("sim-periods", "value"),
                State("config-json", "value"))
def run_sim(n_clicks, sim_periods, config_json):
    if not n_clicks:
        raise PreventUpdate

    try:
        sim = build_simulation_from_config(sim_periods, config_json)
        sim = sim.run()
        df = get_results(sim)

        # pick inventory columns to plot (columns that end with 'inventory')
        inv_cols = [c for c in df.columns if c.endswith(" inventory")]
        fig = {
            "data": [],
            "layout": {"title": "Inventories over time", "xaxis": {"title": "Period"}},
        }
        for col in inv_cols:
            fig["data"].append({"x": list(range(len(df))), "y": df[col].tolist(), "name": col.replace(" inventory", "")})

        return (
            f"Simulation complete. Generated {len(df)} rows of results.",
            df.to_json(date_format="iso", orient="split"),
            fig,
        )

    except Exception as e:
        tb = traceback.format_exc()
        return f"Error running simulation:\n{e}\n{tb}", None, {"data": [], "layout": {"title": "Error"}}


@app.callback(
    Output("download", "data"),
    Input("download-button", "n_clicks"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def download_results(n_clicks, results_json):
    if not results_json:
        raise PreventUpdate
    try:
        df = pd.read_json(results_json, orient="split")

        def to_excel_bytes():
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            buffer.seek(0)
            return buffer.read()

        return dcc.send_bytes(to_excel_bytes, filename="simulation_results.xlsx")

    except Exception as e:
        raise


if __name__ == "__main__":
    # Dash v2+ replaced app.run_server with app.run
    # Use app.run(...) to start the server (keeps debug behaviour)
    app.run(debug=True)
