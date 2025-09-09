import random
import matplotlib.pyplot as plt
import numpy as np
import csv

# -----------------------------
# JAM Spec Parameters (from Glossário and Gray Paper v0.7.1)
# -----------------------------
C = 341                 # Number of cores
P = 6                   # Slot time (s)
T = 128                 # Extrinsics per work-package
core_bandwidth = 2 * 1024 * 1024  # 2 MB/s per core
work_package_size = 13_794_305    # Bytes (WB)
witness_size = 640      # Bytes per witness
finality_delay_slots = 2 # GRANDPA finality delay is ~1-2 blocks. A block is one slot.

# Prices (USD placeholders for pending economy)
GAMMA_A = 0.001   # Cost per workload
GAMMA_Z = 0.0005  # Cost per ticket

# -----------------------------
# Simulation Function
# -----------------------------
def run_simulation(num_slots=20, scenario="mixed", epoch_mode=False):
    """
    Run a JAM workload simulation with per-core distribution and priority.
    This version models contention on a per-core basis and includes finality.
    
    Args:
        num_slots (int): Slots to simulate (set to 600 for epoch_mode=True).
        scenario (str): 'stateless', 'state-heavy', or 'mixed'.
        epoch_mode (bool): Simulate full epoch (E=600 slots ~1h) with entropy reseed.
    
    Returns:
        dict: Results including TPS, costs, and contention rates.
    """
    if epoch_mode:
        num_slots = 600  # E=600 slots for 1h epoch
    priorities = np.linspace(0.3, 0.7, num_slots)
    contention_history = []
    costs = []
    tps_effective = []
    tps_finalized = []

    for slot in range(num_slots):
        priority = priorities[slot]
        if epoch_mode and slot % 100 == 0:
            random.seed(random.randint(0, 2**32 - 1))

        if scenario == "stateless":
            workloads = random.randint(1, 3)
            witnesses = random.randint(100, 500)
        elif scenario == "state-heavy":
            workloads = random.randint(3, 6)
            witnesses = random.randint(1000, 3000)
        elif scenario == "mixed":
            if random.random() < 0.8:
                workloads = random.randint(1, 3)
                witnesses = random.randint(100, 500)
            else:
                workloads = random.randint(3, 6)
                witnesses = random.randint(1000, 3000)
        else:
            raise ValueError("Unknown scenario.")

        workloads_per_core = np.random.multinomial(workloads, [1/C]*C)
        witnesses_per_core = np.random.multinomial(witnesses, [1/C]*C)

        slot_contention_count = 0
        total_processed_data_in_slot = 0
        
        for i in range(C):
            wp_data_per_core = workloads_per_core[i] * work_package_size
            wit_data_per_core = witnesses_per_core[i] * witness_size
            total_data_rate_per_core = (wp_data_per_core + wit_data_per_core) / P

            if total_data_rate_per_core > core_bandwidth:
                slot_contention_count += 1
            
            # Calculate total processed data for this core, limited by bandwidth
            processed_data_per_core = min(core_bandwidth * P, wp_data_per_core + wit_data_per_core)
            total_processed_data_in_slot += processed_data_per_core

        contention_history.append(slot_contention_count > 0)
        
        # Calculate effective TPS based on total processed data across all cores
        total_extrinsics_processed = (total_processed_data_in_slot / work_package_size) * T
        effective_tps = total_extrinsics_processed / P
        tps_effective.append(effective_tps)
        
        # --- NEW: GRANDPA Finality Model ---
        # We can only consider transactions finalized after the delay
        if slot >= finality_delay_slots:
            tps_finalized.append(tps_effective[slot - finality_delay_slots])
        else:
            tps_finalized.append(0) # Not finalized yet

        tickets = workloads * T
        cost = GAMMA_A * workloads + GAMMA_Z * tickets
        costs.append(cost)

    tps_theoretical = C * T / P
    avg_tps_effective = np.mean(tps_effective)
    avg_tps_finalized = np.mean(tps_finalized)
    avg_cost = np.mean(costs)
    contention_rate = sum(contention_history) / num_slots * 100

    with open('simulation_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Slot', 'Effective TPS', 'Finalized TPS', 'Contention'])
        for slot in range(num_slots):
            writer.writerow([slot, tps_effective[slot], tps_finalized[slot], contention_history[slot]])

    return {
        "scenario": scenario,
        "tps_theoretical": tps_theoretical,
        "avg_tps_effective": avg_tps_effective,
        "avg_tps_finalized": avg_tps_finalized,
        "avg_cost": avg_cost,
        "contention_rate": contention_rate,
        "tps_effective": tps_effective,
        "tps_finalized": tps_finalized,
        "contention_history": contention_history
    }

# -----------------------------
# Run the 3 scenarios
# -----------------------------
results = []
for scenario in ["stateless", "state-heavy", "mixed"]:
    res = run_simulation(num_slots=20, scenario=scenario, epoch_mode=False)
    results.append(res)
    print(f"\n--- Scenario: {scenario} ---")
    print(f"Theoretical throughput: {res['tps_theoretical']:.2f} TPS")
    print(f"Average effective throughput: {res['avg_tps_effective']:.2f} TPS")
    print(f"Average finalized throughput: {res['avg_tps_finalized']:.2f} TPS")
    print(f"Average contention rate: {res['contention_rate']:.1f}% of slots")
    print(f"Average cost per slot: ${res['avg_cost']:.4f}")

# -----------------------------
# Comparative visualization
# -----------------------------
plt.figure(figsize=(10, 6))
for res in results:
    plt.plot(res["tps_effective"], linestyle='--', marker='o', label=f"{res['scenario']} (Effective TPS)")
    plt.plot(res["tps_finalized"], marker='x', label=f"{res['scenario']} (Finalized TPS)")
plt.axhline(y=results[0]["tps_theoretical"], color='r', linestyle='--', label='Theoretical TPS')
plt.xlabel('Slot')
plt.ylabel('TPS')
plt.title('JAM Simulation: Workload Scenarios with Finality')
plt.legend()
plt.tight_layout()
plt.show()
