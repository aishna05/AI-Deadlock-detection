import numpy as np
import pandas as pd
import random

def is_safe_bankers_algo(available, allocation, need):
    """
    Implements the Banker's Safety Algorithm to check if a system state is safe.
    
    A state is safe if a sequence of process executions exists that allows all 
    processes to complete without causing a deadlock.

    Args:
        available (np.ndarray): 1D array of available resources.
        allocation (np.ndarray): 2D array of resources allocated to each process.
        need (np.ndarray): 2D array of the remaining resource needs of each process.

    Returns:
        bool: True if the state is safe, False otherwise.
    """
    num_processes, num_resources = allocation.shape
    work = available.copy()
    finish = np.array([False] * num_processes)
    safe_sequence = []   # ✅ initialize safe sequence list

    while len(safe_sequence) < num_processes:
        found_process = False
        for i in range(num_processes):
            # Find a process that is not yet finished and whose needs can be met
            if not finish[i] and all(need[i] <= work):
                # Simulate executing this process
                work += allocation[i]
                finish[i] = True
                safe_sequence.append(i)
                found_process = True
        
        if not found_process:
            return False  # unsafe state
    
    return True  # safe state


def generate_state(num_processes, num_resources, total_resources):
    """
    Generates a random, valid operating system state for deadlock analysis.
    """
    while True:
        # Random Max demand and Allocation
        max_matrix = np.random.randint(0, total_resources + 1, size=(num_processes, num_resources))
        allocation_matrix = np.random.randint(0, max_matrix + 1)

        # Calculate Available = Total - Allocated
        allocated_sum = np.sum(allocation_matrix, axis=0)
        available_vector = total_resources - allocated_sum

        if np.all(available_vector >= 0):  # valid state
            need_matrix = max_matrix - allocation_matrix
            return available_vector, allocation_matrix, need_matrix


def create_dataset(num_samples, num_processes, num_resources, total_resources, balance=False):
    """
    Generates a labeled synthetic dataset.
    """
    data = []   # ✅ initialize list
    labels = [] # ✅ initialize list
    unsafe_count = 0
    
    print(f"Generating dataset with {num_samples} samples...")
    while len(data) < num_samples:
        available, allocation, need = generate_state(num_processes, num_resources, total_resources)
        is_safe = is_safe_bankers_algo(available, allocation, need)
        
        # Balance dataset if required
        if balance and is_safe and unsafe_count < num_samples / 2:
            continue
            
        features = np.concatenate([
            available.flatten(), 
            allocation.flatten(), 
            need.flatten()
        ])
        
        data.append(features)
        label = 1 if is_safe else 0
        labels.append(label)
        
        if not is_safe:
            unsafe_count += 1
            
        if (len(data) % 1000) == 0:
            print(f"Generated {len(data)} samples...")

    df = pd.DataFrame(data)
    df['label'] = labels
    return df


if __name__ == "__main__":
    # Parameters
    NUM_PROCESSES = 5
    NUM_RESOURCES = 3
    TOTAL_RESOURCES = np.array([1, 2, 3])  # Example
    NUM_SAMPLES = 5000  # reduced for demo (50000 is heavy)

    # Generate dataset
    dataset = create_dataset(
        num_samples=NUM_SAMPLES,
        num_processes=NUM_PROCESSES,
        num_resources=NUM_RESOURCES,
        total_resources=TOTAL_RESOURCES,
        balance=True
    )
    
    # Print summary
    print("\n--- Dataset Generation Complete ---")
    print(f"Total samples: {len(dataset)}")
    print(f"Distribution of labels:\n{dataset['label'].value_counts()}")
    print("\nFirst 5 rows of the generated DataFrame:")
    print(dataset.head())
