# Banker's Algorithm Implementation in Python
# This is a simulation of the Banker's algorithm for deadlock avoidance in operating systems.
# It checks if granting a resource request to a process will leave the system in a safe state.
# A safe state means there exists a sequence of processes that can complete without deadlock.

# Key Matrices/Vectors:
# - Allocation: Current resources allocated to each process (n x m matrix, n=processes, m=resource types)
# - Max: Maximum resources each process may need (n x m matrix)
# - Available: Currently available resources (1 x m vector)
# - Need: Resources still needed by each process (Need = Max - Allocation, n x m matrix)

# The algorithm includes:
# 1. Request validation: Check if request <= Need and request <= Available.
# 2. Simulate allocation: Temporarily allocate resources and update Available, Allocation, Need.
# 3. Safety check: Use a work vector and finish array to find a safe sequence.

# Example usage is provided at the end.

def is_safe(available, max_demand, allocation):
    """
    Safety algorithm to check if the system is in a safe state.
    :param available: List of available resources
    :param max_demand: 2D list of max demand per process
    :param allocation: 2D list of current allocation per process
    :return: True if safe, False otherwise
    """
    num_processes = len(max_demand)
    num_resources = len(available)
    
    # Calculate Need matrix: Need[i][j] = Max[i][j] - Allocation[i][j]
    need = [[max_demand[i][j] - allocation[i][j] for j in range(num_resources)] for i in range(num_processes)]
    
    # Work vector: Copy of available resources
    work = available[:]
    
    # Finish array: False initially for all processes
    finish = [False] * num_processes
    
    # Safe sequence list
    safe_sequence = []
    
    # Loop to find processes that can be finished
    while len(safe_sequence) < num_processes:
        found = False
        for p in range(num_processes):
            if not finish[p] and all(need[p][j] <= work[j] for j in range(num_resources)):
                # Simulate allocation: Add allocated resources back to work
                work = [work[j] + allocation[p][j] for j in range(num_resources)]
                finish[p] = True
                safe_sequence.append(p)
                found = True
                break
        if not found:
            return False, []  # No safe sequence found
    return True, safe_sequence

def bankers_algorithm(available, max_demand, allocation, request, process_id):
    """
    Banker's algorithm to handle a resource request.
    :param available: List of available resources
    :param max_demand: 2D list of max demand per process
    :param allocation: 2D list of current allocation per process
    :param request: List of requested resources for the process
    :param process_id: ID of the requesting process (0-based index)
    :return: (Granted: True/False, Safe Sequence if granted)
    """
    num_resources = len(available)
    
    # Step 1: Check if request <= Need
    need = [max_demand[process_id][j] - allocation[process_id][j] for j in range(num_resources)]
    if not all(request[j] <= need[j] for j in range(num_resources)):
        return False, [], "Request exceeds maximum need"
    
    # Step 2: Check if request <= Available
    if not all(request[j] <= available[j] for j in range(num_resources)):
        return False, [], "Request exceeds available resources"
    
    # Step 3: Simulate allocation
    # Temporarily update Available, Allocation, Need
    temp_available = [available[j] - request[j] for j in range(num_resources)]
    temp_allocation = [allocation[process_id][j] + request[j] for j in range(num_resources)]
    temp_max_demand = max_demand[:]  # Max doesn't change
    temp_allocation_full = allocation[:]
    temp_allocation_full[process_id] = temp_allocation
    
    # Step 4: Check safety
    safe, safe_sequence = is_safe(temp_available, temp_max_demand, temp_allocation_full)
    if safe:
        return True, safe_sequence, "Request granted. Safe sequence: " + str(safe_sequence)
    else:
        return False, [], "Request denied: Would lead to unsafe state"

# Example Usage
if __name__ == "__main__":
    # Example data (from Tanenbaum's Modern Operating Systems or similar)
    # 5 processes (P0 to P4), 3 resource types (A, B, C)
    max_demand = [
        [7, 5, 3],  # P0
        [3, 2, 2],  # P1
        [9, 0, 2],  # P2
        [2, 2, 2],  # P3
        [4, 3, 3]   # P4
    ]
    allocation = [
        [0, 1, 0],  # P0
        [2, 0, 0],  # P1
        [3, 0, 2],  # P2
        [2, 1, 1],  # P3
        [0, 0, 2]   # P4
    ]
    available = [3, 3, 2]  # Available resources
    
    # Example request: Process P1 requests [1, 0, 2]
    process_id = 1  # P1
    request = [1, 0, 2]
    
    granted, safe_sequence, message = bankers_algorithm(available, max_demand, allocation, request, process_id)
    print(message)
    
    # Another request: Process P4 requests [3, 3, 0] (should be denied in this state)
    process_id = 4  # P4
    request = [3, 3, 0]
    granted, safe_sequence, message = bankers_algorithm(available, max_demand, allocation, request, process_id)
    print(message)