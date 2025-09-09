# jam-project
Polkadot JAM

# JAM Simulation

This repository contains a Python-based simulation of the core mechanics of the Join-Accumulate Machine (JAM) protocol, as described in the Gray Paper v0.7.1.

The simulation models key concepts such as:
* **Per-Core Contention:** Simulating how workloads and data are distributed across 341 cores, identifying potential bottlenecks.
* **Epoch-based Entropy:** Modeling the random distribution of tasks to prevent single-core overload over time.
* **Workload Scenarios:** Testing the network's performance under `stateless`, `state-heavy`, and `mixed` conditions.
* **GRANDPA Finality:** Simulating transaction finalization with a realistic block delay.

## Installation

This simulation requires Python 3 and a few libraries.

First, ensure you have Python 3 and `pip` installed. Then, navigate to the project directory and install the dependencies:

```bash
pip install numpy matplotlib

Usage

To run the simulation, simply execute the simulation.py script from your terminal:

python3 scripts/simulation.py


The script will output the simulation results for various scenarios and generate a comparative plot. It also exports the data to a simulation_results.csv file.

License

This project is licensed under the MIT License.


