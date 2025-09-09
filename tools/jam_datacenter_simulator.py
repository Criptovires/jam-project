#!/usr/bin/env python3
"""
Jam‑throughput simulator – data‑center edition
================================================

Purpose
-------
Provide a single‑file, zero‑install (apart from numpy/matplotlib) harness that
can be executed on any test rig – from a single‑CPU dev box up to a
multi‑thousand‑core data‑center server – to evaluate the Jam protocol’s
theoretical and effective TPS under different workload mixes.

How it works
------------
*   The script models a *slot* of length P seconds.
*   Each logical core (C) has a fixed outbound bandwidth (core_bandwidth).
*   A “work‑package” is 13 794 305 bytes (~13 MiB) and contains T = 128
    extrinsics (transactions).
*   For each slot a random number of work‑packages is generated according to
    the selected scenario (stateless, state‑heavy, mixed).
*   Work‑packages are split into 1 MiB chunks so they can be distributed
    across many cores – this matches the way Jam spreads work across the
    core pool.
*   Contention is flagged when a core’s required data rate exceeds its
    allotted bandwidth for the slot.
*   Effective TPS = processed extrinsics / slot‑time.
*   Finalized TPS lags the effective value by `finality_delay_slots`
    (Grandpa finality).

Configuration
-------------
Edit the block marked “=== USER SETTINGS ===” to match the test rig.
All other sections are generic and need not be touched.

Output
------
*   Human‑readable summary printed to stdout.
*   Optional CSV (`jam_results_<scenario>.csv`) containing per‑slot metrics.
*   Optional Matplotlib chart (`jam_plot_<scenario>.png`) visualising
    effective/ finalized TPS and contention over time.

"""

# -------------------------------------------------
# === USER SETTINGS ==========================================================
# ------------------------------------------------------------------------------

# 1️⃣  Core pool size – logical cores that Jam would see.
#     For a single‑CPU dev box you might set C = 16.
#     For a dual‑socket Xeon server you might set C = 96.
#     For the reference Jam spec use C = 341.
#     For a “mega‑Jam” deployment (multiple CoreChains) use a multiple of 341.
C = 341                     # <-- edit as needed

# 2️⃣  Per‑core outbound bandwidth (bytes per second).
#     Typical data‑center NIC: 25 GbE ≈ 3 125 MiB/s total.
#     Distribute evenly across the logical cores you model.
#     Example for a 25 GbE NIC and 341 cores:
core_bandwidth = (3125 * 1024 * 1024) // C   # bytes/s per core
# If you want to force a higher value (e.g., 5 MiB/s) you can replace the line above:
# core_bandwidth = 5 * 1024 * 1024

# 3️⃣  Slot length (seconds). The Jam spec uses 6 s.
P = 6                       # seconds per slot

# 4️⃣  Number of extrinsics per work‑package (fixed by the protocol).
T = 128

# 5️⃣  Size of a full work‑package (bytes). ~13 MiB.
work_package_size = 13_794_305

# 6️⃣  Chunk size used to split a work‑package for distribution.
#     1 MiB works well for most data‑center tests.
CHUNK_SIZE = 1 * 1024 * 1024

# 7️⃣  Witness size (tiny, kept for completeness).
witness_size = 640

# 8️⃣  Finality delay in slots (Grandpa finality).
finality_delay_slots = 2

# 9️⃣  Economic parameters – keep the defaults unless you are doing a cost study.
GAMMA_A = 0.001   # USD per work‑package
GAMMA_Z = 0.0005  # USD per ticket (ticket = T extrinsics)

# 🔟  Which scenario(s) to run.
#     Options: "stateless", "state-heavy", "mixed"
SCENARIOS = ["stateless", "state-heavy", "mixed"]

# 1️⃣1️⃣  Number of slots to simulate per scenario.
#      For a quick sanity check 20–30 slots is enough;
#      for a statistically robust run use 200–500.
NUM_SLOTS = 200

# 1️⃣2️⃣  CSV export toggle – set to True if you want a detailed dump.
EXPORT_CSV = True

# 1️⃣3️⃣  Plotting toggle – set to True if matplotlib is available and you want a PNG.
EXPORT_PLOT = True

# -------------------------------------------------
# End of user settings – the rest of the file is generic.
# -------------------------------------------------

import random
import csv
import numpy as np
import sys

if EXPORT_PLOT:
    import matplotlib.pyplot as plt


def run_simulation(num_slots: int, scenario: str):
    """
    Runs the Jam throughput model for a single scenario.

    Returns a dict with aggregated statistics and per‑slot series.
    """
    contention_flags = []          # True if any core overloaded this slot
    effective_tps_series = []     # TPS after processing the slot
    finalized_tps_series = []     # TPS after finality delay
    cost_series = []               # USD cost for the slot

    # Helper: how many workloads (full work‑packages) this slot gets?
    def workloads_for_slot():
        if scenario == "stateless":
            return random.randint(5, 15)   # light, mostly stateless txs
        elif scenario == "state-heavy":
            return random.randint(15, 30)  # many state‑changing ops
        else:  # mixed
            # blend the two ranges – pick a random point in the combined interval
            return random.randint(5, 30)

    for slot_idx in range(num_slots):
        # -------------------- 1️⃣  Generate workload --------------------
        num_workloads = workloads_for_slot()                     # # of full work‑packages
        total_chunks = num_workloads * (work_package_size // CHUNK_SIZE)

        # Distribute chunks uniformly across the logical core pool
        chunks_per_core = np.random.multinomial(
            total_chunks, [1 / C] * C
        )  # length C array, sum = total_chunks

        # -------------------- 2️⃣  Bandwidth check --------------------
        slot_contended = False
        processed_bytes = 0

        for core_idx in range(C):
            # Data this core *needs* to push this slot:
            #   chunks * CHUNK_SIZE  +  (optional) witness bytes
            #   (we add a tiny witness per workload that lands on the core)
            witness_bytes = (num_workloads // C) * witness_size
            needed = chunks_per_core[core_idx] * CHUNK_SIZE + witness_bytes

            # Capacity of this core for the slot:
            capacity = core_bandwidth * P

            if needed > capacity:
                slot_contended = True

            # Count only what actually gets transmitted (capacity caps it)
            processed_bytes += min(needed, capacity)

        contention_flags.append(slot_contended)

        # -------------------- 3️⃣  TPS calculation --------------------
        # How many extrinsics were *actually* processed this slot?
        extrinsics_processed = (processed_bytes / work_package_size) * T
        effective_tps = extrinsics_processed / P
        effective_tps_series.append(effective_tps)

        # Finalized TPS lags by `finality_delay_slots`
        if slot_idx >= finality_delay_slots:
            finalized_tps_series.append(effective_tps_series[slot_idx - finality_delay_slots])
        else:
            finalized_tps_series.append(0.0)

        # -------------------- 4️⃣  Cost calculation --------------------
        tickets = num_workloads * T
        slot_cost = GAMMA_A * num_workloads + GAMMA_Z * tickets
        cost_series.append(slot_cost)

    # -------------------- 5️⃣  Aggregate results --------------------
    theoretical_tps = C * T / P

    results = {
        "scenario": scenario,
        "theoretical_tps": theoretical_tps,
        "avg_effective_tps": float(np.mean(effective_tps_series)),
        "avg_finalized_tps": float(np.mean(finalized_tps_series)),
        "contention_rate_pct": 100.0 * sum(contention_flags) / num_slots,
        "avg_cost_per_slot_usd": float(np.mean(cost_series)),
        "effective_series": effective_tps_series,
        "finalized_series": finalized_tps_series,
        "contention_series": contention_flags,
        "cost_series": cost_series,
    }

    return results


def export_csv(results: dict):
    """Write per‑slot data to a CSV file."""
    filename = f"jam_results_{results['scenario']}.csv"
    header = [
        "slot_index",
        "effective_tps",
        "finalized_tps",
        "contended (bool)",
        "slot_cost_usd",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(results["effective_series"])):
            writer.writerow(
                [
                    i,
                    results["effective_series"][i],
                    results["finalized_series"][i],
                    results["contention_series"][i],
                    results["cost_series"][i],
                ]
            )
    print(f"[+] CSV exported → {filename}")


def export_plot(results: dict):
    """Create a PNG chart showing TPS and contention."""
    filename = f"jam_plot_{results['scenario']}.png"
    slots = np.arange(len(results["effective_series"]))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # TPS lines (effective & finalized)
    ax1.plot(slots, results["effective_series"], label="Effective TPS", color="#1f77b4")
    ax1.plot(
        slots,
        results["finalized_series"],
        label="Finalized TPS",
        color="#ff7f0e",
        linestyle="--",
    )
    ax1.set_xlabel("Slot index")
    ax1.set_ylabel("TPS")
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left")

    # Contention as a secondary axis (percentage of slots)
    ax2 = ax1.twinx()
    contention_pct = np.cumsum(results["contention_series"]) / (slots + 1) * 100
    ax2.plot(
        slots,
        contention_pct,
        label="Cumulative contention %", color="#2ca02c", linewidth=1.5
    )
    ax2.set_ylabel("Cumulative contention (%)")
    ax2.tick_params(axis="y")
    ax2.legend(loc="upper right")

    plt.title(f"Jam throughput – {results['scenario'].replace('-', ' ').title()}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[+] Plot exported → {filename}")


def main():
    print("\n=== Jam Throughput Simulator – Data‑Center Edition ===\n")
    print(f"Configuration:")
    print(f"  • Logical cores (C)                : {C}")
    print(f"  • Per‑core bandwidth (MiB/s)       : {core_bandwidth / (1024*1024):.2f}")
    print(f"  • Slot length (seconds)            : {P}")
    print(f"  • Work‑package size (MiB)          : {work_package_size / (1024*1024):.2f}")
    print(f"  • Chunk size (MiB)                 : {CHUNK_SIZE / (1024*1024)}")
    print(f"  • Finality delay (slots)           : {finality_delay_slots}")
    print(f"  • Scenarios to run                : {', '.join(SCENARIOS)}")
    print(f"  • Slots per scenario               : {NUM_SLOTS}\n")

    for scen in SCENARIOS:
        print(f"▶️  Running scenario: {scen}")
        res = run_simulation(NUM_SLOTS, scen)

        print("\n--- Summary ---------------------------------------------------")
        print(f"Scenario                : {res['scenario']}")
        print(f"Theoretical TPS         : {res['theoretical_tps']:.2f}")
        print(f"Avg. effective TPS      : {res['avg_effective_tps']:.2f}")
        print(f"Avg. finalized TPS      : {res['avg_finalized_tps']:.2f}")
        print(f"Contention rate         : {res['contention_rate_pct']:.1f}% of slots")
        print(f"Avg. cost per slot (USD): ${res['avg_cost_per_slot_usd']:.4f}")
        print("---------------------------------------------------------------\n")

        if EXPORT_CSV:
            export_csv(res)

        if EXPORT_PLOT:
            export_plot(res)


if __name__ == "__main__":
    # Guard against accidental execution on a machine with too few cores.
    # If you really want to run on a tiny dev box, set the env var
    #   JAM_ALLOW_SMALL=1
    import os
    min_cores_needed = 64
    if C < min_cores_needed and os.getenv("JAM_ALLOW_SMALL") != "1":
        sys.stderr.write(
            f"\n[!] Warning: You have configured C={C} which is far below a typical "
            f"data‑center deployment (>= {min_cores_needed} logical cores).\n"
            "If you *really* intend to run on a small box, export the env var:\n"
            "    export JAM_ALLOW_SMALL=1\n"
            "and re‑execute the script.\n"
        )
        sys.exit(1)

    main()
