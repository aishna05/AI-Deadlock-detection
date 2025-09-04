# Resource Allocation Graph (RAG) Algorithm for Deadlock Detection
# This script implements a RAG-based deadlock detection algorithm in Python.
# It uses a graph structure to model processes and resources, detects cycles to identify deadlocks,
# and can be integrated with the AI-Powered Deadlock Detection System.

# Key Components:
# - Graph: Dictionary representing the RAG (processes and resources as nodes, request/assignment edges).
# - Cycle Detection: Depth-First Search (DFS) to find cycles indicating deadlocks.
# - Integration: Designed to work with Psutil for real-time data and FastAPI for API exposure.

# Example usage is provided at the end.

def detect_cycles(graph, node, visited, rec_stack, path, cycles):
    """
    Detect cycles in the RAG using DFS.
    :param graph: Dictionary representing the RAG (node -> list of neighbors).
    :param node: Current node (process or resource).
    :param visited: Set of visited nodes.
    :param rec_stack: Set of nodes in the current recursion stack.
    :param path: List tracking the current path for cycle reporting.
    :param cycles: List to store detected cycles.
    :return: None (cycles are appended to the cycles list).
    """
    visited.add(node)
    rec_stack.add(node)
    path.append(node)
    
    # Explore neighbors (processes or resources)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            detect_cycles(graph, neighbor, visited, rec_stack, path, cycles)
        elif neighbor in rec_stack:
            # Cycle found: Extract the cycle from the path
            cycle_start = path.index(neighbor)
            cycles.append(path[cycle_start:])
    
    rec_stack.remove(node)
    path.pop()

def rag_deadlock_detection(processes, resources, request_edges, assignment_edges):
    """
    Main RAG algorithm to detect deadlocks.
    :param processes: List of process IDs (e.g., ['P1', 'P2']).
    :param resources: List of resource IDs (e.g., ['R1', 'R2']).
    :param request_edges: List of tuples (process, resource) for request edges.
    :param assignment_edges: List of tuples (resource, process) for assignment edges.
    :return: List of cycles (each cycle is a list of nodes indicating a deadlock).
    """
    # Construct the RAG as an adjacency list
    graph = {}
    
    # Initialize graph with processes and resources
    for p in processes:
        graph[p] = []
    for r in resources:
        graph[r] = []
    
    # Add request edges (Process -> Resource)
    for p, r in request_edges:
        graph[p].append(r)
    
    # Add assignment edges (Resource -> Process)
    for r, p in assignment_edges:
        graph[r].append(p)
    
    # Detect cycles using DFS
    visited = set()
    rec_stack = set()
    cycles = []
    path = []
    
    # Check for cycles starting from each process
    for node in processes:
        if node not in visited:
            detect_cycles(graph, node, visited, rec_stack, path, cycles)
    
    return cycles

# Example Usage
if __name__ == "__main__":
    # Example RAG:
    # P1 holds R1, requests R2
    # P2 holds R2, requests R1
    # This creates a cycle: P1 -> R2 -> P2 -> R1 -> P1
    processes = ['P1', 'P2']
    resources = ['R1', 'R2']
    request_edges = [('P1', 'R2'), ('P2', 'R1')]
    assignment_edges = [('R1', 'P1'), ('R2', 'P2')]
    
    # Run the RAG algorithm
    cycles = rag_deadlock_detection(processes, resources, request_edges, assignment_edges)
    
    if cycles:
        print("Deadlocks detected. Cycles found:")
        for cycle in cycles:
            print(f"Cycle: {' -> '.join(cycle)}")
    else:
        print("No deadlocks detected.")