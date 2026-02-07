import struct
import pickle
import numpy as np
from pathlib import Path


def read_index_file(filepath):
    """
    Read MSAG index file.

    Returns:
        List of lists, where each inner list contains neighbor IDs for a node
    """
    graph = []

    with open(filepath, 'rb') as f:
        while True:
            # Read number of neighbors
            data = f.read(4)
            if not data:
                break

            gk = struct.unpack('I', data)[0]  # 'I' = unsigned int

            # Read neighbor IDs
            neighbor_data = f.read(gk * 4)
            if len(neighbor_data) < gk * 4:
                break

            neighbors = struct.unpack(f'{gk}I', neighbor_data)
            graph.append(list(neighbors))

    return graph


def merge_indices(base_path, prefix="celeba_MSAG.index", output_file="merged_indices.pkl",
                  save_binary=True, binary_output_file="merged_indices.bin"):
    """
    Merge index files from index_1_0 down to index_0_1.

    Each row becomes a list of lists where:
    - First list: values from index_1_0
    - Second list: NEW values from index_0.9_0.1 not in first list
    - Third list: NEW values from index_0.8_0.2 not in previous lists
    - And so on...

    Args:
        base_path: Directory containing the index files
        prefix: Prefix of the index files
        output_file: Output filename for the merged result (pickle format)
        save_binary: If True, also save in binary format
        binary_output_file: Output filename for binary format
    """
    base_path = Path(base_path)

    # Define the order of files to process
    file_suffixes = [
        "1_0",
        "0.9_0.1",
        "0.8_0.2",
        "0.7_0.3",
        "0.6_0.4",
        "0.5_0.5",
        "0.4_0.6",
        "0.3_0.7",
        "0.2_0.8",
        "0.1_0.9",
        "0_1"
    ]

    file_paths = [base_path / f"{prefix}_{suffix}" for suffix in file_suffixes]

    # Check which files exist
    existing_files = []
    for fpath in file_paths:
        if fpath.exists():
            existing_files.append(fpath)
        else:
            print(f"Warning: File not found: {fpath}")

    if not existing_files:
        print("Error: No index files found!")
        return None

    print(f"Found {len(existing_files)} index files to merge")
    print("="*60)

    # Read the first file to initialize
    print(f"Reading base file: {existing_files[0].name}")
    merged_graph = []
    first_graph = read_index_file(existing_files[0])
    num_nodes = len(first_graph)

    # Initialize merged graph with the first file's data
    for neighbors in first_graph:
        merged_graph.append([neighbors])  # Each row is a list of lists

    print(f"  - Loaded {num_nodes} nodes")

    # Process remaining files
    for i, fpath in enumerate(existing_files[1:], start=1):
        print(f"\nMerging file {i+1}/{len(existing_files)}: {fpath.name}")
        current_graph = read_index_file(fpath)

        if len(current_graph) != num_nodes:
            print(f"  Warning: Node count mismatch! Expected {num_nodes}, got {len(current_graph)}")
            num_nodes = min(num_nodes, len(current_graph))

        new_values_count = 0

        # For each node, find new values
        for node_idx in range(num_nodes):
            # Get all values seen so far for this node
            seen_values = set()
            for prev_list in merged_graph[node_idx]:
                seen_values.update(prev_list)

            # Find new values in current file
            current_neighbors = current_graph[node_idx]
            new_neighbors = [val for val in current_neighbors if val not in seen_values]

            # Add the new values as a new list
            merged_graph[node_idx].append(new_neighbors)
            new_values_count += len(new_neighbors)

        print(f"  - Added {new_values_count} new neighbor values across all nodes")

    print("\n" + "="*60)
    print(f"Merge complete! Total nodes: {len(merged_graph)}")

    # Calculate statistics
    total_lists = sum(len(node_lists) for node_lists in merged_graph)
    total_values = sum(len(lst) for node_lists in merged_graph for lst in node_lists)

    print(f"Total neighbor lists: {total_lists}")
    print(f"Total neighbor values: {total_values}")
    print(f"Average values per node: {total_values / len(merged_graph):.2f}")

    # Save the merged result in pickle format
    output_path = base_path / output_file
    with open(output_path, 'wb') as f:
        pickle.dump(merged_graph, f)

    print(f"\nMerged index saved (pickle) to: {output_path}")

    # Also save in binary format if requested
    if save_binary:
        binary_path = base_path / binary_output_file
        save_merged_indices_binary(merged_graph, binary_path)

    return merged_graph


def save_merged_indices_binary(merged_graph, output_path):
    """
    Save merged indices in binary format (similar to original index files).

    Format for each node:
    - num_sublists (4 bytes): number of lists in the list-of-lists
    - For each sublist:
        - num_neighbors (4 bytes): number of neighbors in this sublist
        - neighbor_ids (num_neighbors * 4 bytes): the neighbor IDs

    Args:
        merged_graph: List of list-of-lists structure
        output_path: Path to save the binary file
    """
    with open(output_path, 'wb') as f:
        for node_lists in merged_graph:
            # Write number of sublists
            num_sublists = len(node_lists)
            f.write(struct.pack('I', num_sublists))

            # Write each sublist
            for neighbors in node_lists:
                # Write number of neighbors in this sublist
                num_neighbors = len(neighbors)
                f.write(struct.pack('I', num_neighbors))

                # Write the neighbor IDs
                if num_neighbors > 0:
                    f.write(struct.pack(f'{num_neighbors}I', *neighbors))

    print(f"Binary merged index saved to: {output_path}")


def load_merged_indices_binary(filepath):
    """
    Load merged indices from binary format.

    Returns:
        List of list-of-lists structure
    """
    merged_graph = []

    with open(filepath, 'rb') as f:
        while True:
            # Read number of sublists
            data = f.read(4)
            if not data:
                break

            num_sublists = struct.unpack('I', data)[0]
            node_lists = []

            # Read each sublist
            for _ in range(num_sublists):
                # Read number of neighbors in this sublist
                data = f.read(4)
                if not data:
                    break

                num_neighbors = struct.unpack('I', data)[0]

                # Read neighbor IDs
                if num_neighbors > 0:
                    neighbor_data = f.read(num_neighbors * 4)
                    if len(neighbor_data) < num_neighbors * 4:
                        break
                    neighbors = list(struct.unpack(f'{num_neighbors}I', neighbor_data))
                else:
                    neighbors = []

                node_lists.append(neighbors)

            if node_lists:
                merged_graph.append(node_lists)

    return merged_graph


def load_merged_indices(filepath):
    """Load previously saved merged indices (pickle format)."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def analyze_sublist_statistics(merged_graph):
    """
    Analyze statistics for each sublist position across all nodes.
    
    Args:
        merged_graph: List of list-of-lists structure
    """
    if not merged_graph:
        print("Error: Empty graph")
        return
    
    num_sublists = len(merged_graph[0]) if merged_graph else 0
    
    print("\n" + "="*80)
    print("SUBLIST LENGTH STATISTICS ACROSS ALL NODES")
    print("="*80)
    print(f"Total nodes: {len(merged_graph)}")
    print(f"Number of sublists per node: {num_sublists}")
    print("="*80)

    # Header
    print(f"\n{'Sublist':<10} {'Source File':<25} {'Min':<8} {'Max':<8} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Total Values':<12}")
    print("-" * 103)

    file_suffixes = [
        "1_0",
        "0.9_0.1",
        "0.8_0.2",
        "0.7_0.3",
        "0.6_0.4",
        "0.5_0.5",
        "0.4_0.6",
        "0.3_0.7",
        "0.2_0.8",
        "0.1_0.9",
        "0_1"
    ]

    for sublist_idx in range(num_sublists):
        # Collect lengths for this sublist position across all nodes
        lengths = [len(node[sublist_idx]) for node in merged_graph if sublist_idx < len(node)]

        if lengths:
            min_len = np.min(lengths)
            max_len = np.max(lengths)
            mean_len = np.mean(lengths)
            median_len = np.median(lengths)
            std_len = np.std(lengths)
            total_values = np.sum(lengths)

            source = file_suffixes[sublist_idx] if sublist_idx < len(file_suffixes) else f"index_{sublist_idx}"
            source_name = f"index_{source}"

            print(f"{sublist_idx:<10} {source_name:<25} {min_len:<8} {max_len:<8} {mean_len:<10.2f} {median_len:<10.1f} {std_len:<10.2f} {total_values:<12}")

    print("="*80)

    # Additional analysis: Distribution of nodes with empty sublists
    print("\n" + "="*80)
    print("NODES WITH EMPTY SUBLISTS")
    print("="*80)
    print(f"{'Sublist':<10} {'Nodes with 0 values':<25} {'Percentage':<15}")
    print("-" * 50)

    for sublist_idx in range(num_sublists):
        empty_count = sum(1 for node in merged_graph if sublist_idx < len(node) and len(node[sublist_idx]) == 0)
        percentage = (empty_count / len(merged_graph)) * 100
        print(f"{sublist_idx:<10} {empty_count:<25} {percentage:<15.2f}%")

    print("="*80)


def print_example_nodes(merged_graph, num_examples=5):
    """Print example nodes to show the merged structure."""
    print("\n" + "="*60)
    print(f"Example of first {num_examples} nodes:")
    print("="*60)

    for i in range(min(num_examples, len(merged_graph))):
        print(f"\nNode {i}:")
        for j, neighbor_list in enumerate(merged_graph[i]):
            if neighbor_list:  # Only print non-empty lists
                preview = neighbor_list[:10]
                suffix = "..." if len(neighbor_list) > 10 else ""
                print(f"  List {j}: {len(neighbor_list)} values -> {preview}{suffix}")
            else:
                print(f"  List {j}: (empty)")


if __name__ == "__main__":
    # Set the path to the index files
    index_dir = "MUST-main/indexing_and_search/doc/index/celeba"

    print("Starting index merge process...")
    print("="*60)

    # Merge the indices
    merged_graph = merge_indices(index_dir)

    if merged_graph:
        # Print examples
        print_example_nodes(merged_graph, num_examples=5)
        
        # Analyze sublist statistics
        analyze_sublist_statistics(merged_graph)

        print("\n" + "="*60)
        print("To load the merged indices later:")
        print("\nPickle format:")
        print("  from merge_indices import load_merged_indices")
        print("  merged_graph = load_merged_indices('MUST-main/indexing_and_search/doc/index/celeba/merged_indices.pkl')")
        print("\nBinary format:")
        print("  from merge_indices import load_merged_indices_binary")
        print("  merged_graph = load_merged_indices_binary('MUST-main/indexing_and_search/doc/index/celeba/merged_indices.bin')")
