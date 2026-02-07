#!/usr/bin/env python3
"""Test script to verify binary format loading"""

from merge_indices import load_merged_indices_binary
import pickle

# Load the binary format
print("Loading binary format...")
binary_graph = load_merged_indices_binary('MUST-main/indexing_and_search/doc/index/celeba/merged_indices.bin')

# Load the pickle format for comparison
print("Loading pickle format...")
with open('MUST-main/indexing_and_search/doc/index/celeba/merged_indices.pkl', 'rb') as f:
    pickle_graph = pickle.load(f)

print(f"\nBinary format: {len(binary_graph)} nodes")
print(f"Pickle format: {len(pickle_graph)} nodes")

# Compare first 5 nodes
print("\nComparing first 5 nodes...")
all_match = True
for i in range(5):
    if binary_graph[i] == pickle_graph[i]:
        print(f"✓ Node {i}: MATCH")
    else:
        print(f"✗ Node {i}: MISMATCH")
        all_match = False

if all_match:
    print("\n✓ Binary format loads correctly and matches pickle format!")
else:
    print("\n✗ There are mismatches between formats")

# Show example structure
print("\n" + "="*60)
print("Example Node 0 structure (from binary format):")
print("="*60)
for j, sublist in enumerate(binary_graph[0]):
    print(f"List {j}: {len(sublist)} values -> {sublist[:5]}...")
