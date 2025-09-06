import numpy as np
import pandas as pd
from banker import is_safe
from RAG import rag_deadlock_detection  # Assuming RAG file renamed to rag_module.py

def generate_state(num_processes, num_resources, total_resources):
    """Generate a random valid OS state (Available, Allocation, Need)."""
    while True:
        max_matrix = np.random.randint(0, total_resources + 1, size=(num_processes, num_resources))
        allocation_matrix = np.random.randint(0, max_matrix + 1)

        allocated_sum = np.sum(allocation_matrix, axis=0)
        available_vector = total_resources - allocated_sum

        if np.all(available_vector >= 0):
            need_matrix = max_matrix - allocation_matrix
            return available_vector, allocation_matrix, need_matrix

def label_banker(available, max_matrix, allocation_matrix):
    """Label dataset using Banker's algorithm (1=safe, 0=unsafe)."""
    safe, _ = is_safe(available.tolist(), (allocation_matrix + (max_matrix - allocation_matrix)).tolist(), allocation_matrix.tolist())
    return 1 if safe else 0

def label_rag(allocation_matrix, need_matrix=None):
    """Label dataset using RAG algorithm (1=safe, 0=deadlock)."""
    num_processes, num_resources = allocation_matrix.shape
    processes = [f"P{i}" for i in range(num_processes)]
    resources = [f"R{j}" for j in range(num_resources)]

    request_edges = []
    assignment_edges = []

    # Construct RAG edges
    for i in range(num_processes):
        for j in range(num_resources):
            if allocation_matrix[i][j] > 0:
                assignment_edges.append((f"R{j}", f"P{i}"))
            if need_matrix is not None and need_matrix[i][j] > 0:
                request_edges.append((f"P{i}", f"R{j}"))

    cycles = rag_deadlock_detection(processes, resources, request_edges, assignment_edges)
    return 0 if cycles else 1  # 0=deadlock, 1=safe

def create_headers(num_processes, num_resources):
    """Create descriptive headers for CSV."""
    headers = []

    # Available resources
    for r in range(num_resources):
        headers.append(f"Available_R{r}")

    # Allocation matrix
    for p in range(num_processes):
        for r in range(num_resources):
            headers.append(f"P{p}_Alloc_R{r}")

    # Need matrix
    for p in range(num_processes):
        for r in range(num_resources):
            headers.append(f"P{p}_Need_R{r}")

    # Label
    headers.append("label")
    return headers

def create_dataset(num_samples, num_processes, num_resources, total_resources, balance=False, method="banker"):
    """Generate dataset using Banker or RAG with descriptive headers."""
    data = []
    labels = []
    unsafe_count = 0

    print(f"Generating {num_samples} samples using {method} method...")
    while len(data) < num_samples:
        available, allocation, need = generate_state(num_processes, num_resources, total_resources)

        # Label sample
        if method == "banker":
            label = label_banker(available, allocation + need, allocation)
        elif method == "rag":
            label = label_rag(allocation, need)
        else:
            raise ValueError("Method must be 'banker' or 'rag'")

        # Balance dataset if requested
        if balance and label == 1 and unsafe_count < num_samples / 2:
            continue

        # Flatten features: Available + Allocation + Need
        features = np.concatenate([available.flatten(), allocation.flatten(), need.flatten()])
        data.append(features)
        labels.append(label)
        if label == 0:
            unsafe_count += 1

        if len(data) % 1000 == 0:
            print(f"Generated {len(data)} samples...")

    # Create DataFrame with headers
    headers = create_headers(num_processes, num_resources)
    df = pd.DataFrame(np.column_stack([data, labels]), columns=headers)
    return df

if __name__ == "__main__":
    # Example parameters
    NUM_PROCESSES = 5
    NUM_RESOURCES = 3
    TOTAL_RESOURCES = np.array([10, 5, 7])
    NUM_SAMPLES = 5000

    # Generate dataset using Banker's algorithm
    dataset_banker = create_dataset(NUM_SAMPLES, NUM_PROCESSES, NUM_RESOURCES, TOTAL_RESOURCES, balance=True, method="banker")
    print("\nBanker's dataset label distribution:\n", dataset_banker['label'].value_counts())
    dataset_banker.to_csv("deadlock_dataset_banker.csv", index=False)

    # Generate dataset using RAG algorithm
    dataset_rag = create_dataset(NUM_SAMPLES, NUM_PROCESSES, NUM_RESOURCES, TOTAL_RESOURCES, balance=True, method="rag")
    print("\nRAG dataset label distribution:\n", dataset_rag['label'].value_counts())
    dataset_rag.to_csv("deadlock_dataset_rag.csv", index=False)

    print("\nDatasets saved successfully with descriptive headers!")
