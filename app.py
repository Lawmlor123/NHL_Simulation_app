# app.py
from flask import Flask, request, jsonify
from simulation import run_simulation

app = Flask(__name__)

@app.route("/")
def home():
    return "🏒 Hockey Simulation API is running! Try /run-sim"

@app.route("/run-sim")
def run_sim():
    team_a = request.args.get("team_a", "Boston Bruins")
    team_b = request.args.get("team_b", "Toronto Maple Leafs")
    runs = int(request.args.get("runs", 100))

    results = run_simulation(team_a, team_b, runs)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)