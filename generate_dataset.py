"""
generate_dataset.py
AI-Assisted Deadlock Dataset Generator (Banker's + RAG)
Author: Aishna Bhatia
"""

import numpy as np
import pandas as pd
import argparse
from banker import is_safe
from RAG import rag_deadlock_detection


# ==============================
# COMMON UTILITIES
# ==============================
def create_headers(num_processes, num_resources):
    """Generates feature headers for CSV files."""
    headers = [f"Available_R{r}" for r in range(num_resources)]
    for p in range(num_processes):
        headers.extend([f"P{p}_Alloc_R{r}" for r in range(num_resources)])
    for p in range(num_processes):
        headers.extend([f"P{p}_Need_R{r}" for r in range(num_resources)])
    headers.append("label")
    return headers


# ==============================
# BANKER'S DATASET GENERATION
# ==============================
def generate_banker_state(num_processes, num_resources, total_resources):
    """Generates a single random valid state for the Banker's algorithm."""
    while True:
        is_likely_safe = np.random.rand() > 0.6  # ~60% safe states
        max_demand_limit = total_resources // 2 if is_likely_safe else total_resources
        if isinstance(max_demand_limit, (int, float)):
            max_demand_limit = np.full(num_resources, max_demand_limit)

        max_matrix = np.random.randint(0, max_demand_limit + 1, size=(num_processes, num_resources))
        allocation_matrix = np.zeros_like(max_matrix)
        for i in range(num_processes):
            for j in range(num_resources):
                if max_matrix[i, j] > 0:
                    allocation_matrix[i, j] = np.random.randint(0, max_matrix[i, j] + 1)

        allocated_sum = np.sum(allocation_matrix, axis=0)
        available_vector = total_resources - allocated_sum

        if np.all(available_vector >= 0):
            need_matrix = max_matrix - allocation_matrix
            if np.all(need_matrix >= 0):
                return available_vector, allocation_matrix, need_matrix


def create_banker_dataset(num_samples, num_processes, num_resources, total_resources):
    """Generates a labeled dataset for the Banker's Algorithm."""
    print(f"\nGenerating {num_samples} samples for Banker's method...")
    data, labels = [], []

    for _ in range(num_samples):
        available, allocation, need = generate_banker_state(num_processes, num_resources, total_resources)
        safe, _ = is_safe(available.tolist(), (allocation + need).tolist(), allocation.tolist())
        label = 1 if safe else 0
        features = np.concatenate([available.flatten(), allocation.flatten(), need.flatten()])
        data.append(features)
        labels.append(label)

    df = pd.DataFrame(np.column_stack([data, labels]), columns=create_headers(num_processes, num_resources))
    return df


# ==============================
# RAG DATASET GENERATION
# ==============================
def construct_safe_rag_state(num_processes, num_resources, total_resources):
    """Constructs a resource allocation state guaranteed to be cycle-free."""
    allocation = np.zeros((num_processes, num_resources), dtype=int)
    need = np.zeros((num_processes, num_resources), dtype=int)
    available = np.copy(total_resources)

    for p in range(num_processes):
        alloc_req = np.minimum(np.random.randint(0, total_resources // 3 + 1), available)
        allocation[p] = alloc_req
        available -= alloc_req

    procs_with_need = np.random.choice(num_processes, size=max(1, num_processes // 2), replace=False)
    for p in procs_with_need:
        max_possible_need = np.maximum(available, np.zeros_like(available))
        if np.any(max_possible_need > 0):
            need[p] = np.random.randint(0, max_possible_need + 1)
    return available, allocation, need


def construct_unsafe_rag_state(num_processes, num_resources, total_resources):
    """Constructs a resource allocation state with a guaranteed deadlock cycle."""
    allocation = np.zeros((num_processes, num_resources), dtype=int)
    need = np.zeros((num_processes, num_resources), dtype=int)
    available = np.copy(total_resources)

    if num_processes >= 2 and num_resources >= 2:
        alloc_p0_r0 = min(available[0], np.random.randint(1, 5))
        allocation[0, 0] = alloc_p0_r0
        available[0] -= alloc_p0_r0

        alloc_p1_r1 = min(available[1], np.random.randint(1, 5))
        allocation[1, 1] = alloc_p1_r1
        available[1] -= alloc_p1_r1

        need[0, 1] = np.random.randint(1, 5)
        need[1, 0] = np.random.randint(1, 5)

    for p in range(2, num_processes):
        for r in range(num_resources):
            alloc_val = np.random.randint(0, total_resources[r] // 5 + 1)
            allocation[p, r] = min(alloc_val, available[r])
            available[r] -= allocation[p, r]
            need[p, r] = np.random.randint(0, total_resources[r] // 3 + 1)

    return available, allocation, need


def create_rag_dataset(num_samples, num_processes, num_resources, total_resources):
    """Generates a labeled dataset for the RAG method."""
    print(f"\nGenerating {num_samples} samples for RAG method...")
    data, labels = [], []

    for i in range(num_samples):
        if i < num_samples // 2:
            available, allocation, need = construct_safe_rag_state(num_processes, num_resources, total_resources)
            label = 1
        else:
            available, allocation, need = construct_unsafe_rag_state(num_processes, num_resources, total_resources)
            label = 0

        features = np.concatenate([available.flatten(), allocation.flatten(), need.flatten()])
        data.append(features)
        labels.append(label)

    df = pd.DataFrame(np.column_stack([data, labels]), columns=create_headers(num_processes, num_resources))
    return df


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI-assisted deadlock datasets (Banker + RAG).")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples per dataset.")
    parser.add_argument("--processes", type=int, default=5, help="Number of processes.")
    parser.add_argument("--resources", type=int, default=3, help="Number of resource types.")
    args = parser.parse_args()

    NUM_PROCESSES, NUM_RESOURCES, NUM_SAMPLES = args.processes, args.resources, args.samples
    TOTAL_RESOURCES = np.array([20, 15, 20])

    # --- Banker's Dataset ---
    print("\n--- Generating Banker's Dataset ---")
    dataset_banker = create_banker_dataset(NUM_SAMPLES, NUM_PROCESSES, NUM_RESOURCES, TOTAL_RESOURCES)
    print("\nBanker's dataset label distribution:\n", dataset_banker['label'].value_counts(normalize=True))
    dataset_banker.to_csv("deadlock_dataset_banker.csv", index=False)
    print("Banker's dataset saved successfully as 'deadlock_dataset_banker.csv'.")

    # --- RAG Dataset ---
    print("\n--- Generating RAG Dataset ---")
    dataset_rag = create_rag_dataset(NUM_SAMPLES, NUM_PROCESSES, NUM_RESOURCES, TOTAL_RESOURCES)
    print("\nRAG dataset label distribution:\n", dataset_rag['label'].value_counts(normalize=True))
    dataset_rag.to_csv("deadlock_dataset_rag.csv", index=False)
    print("RAG dataset saved successfully as 'deadlock_dataset_rag.csv'.")

    print("\n✅ Both datasets generated and saved successfully!")
