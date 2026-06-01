import json
import sys
import os

# Bootstrap Lár engine path before importing brain modules
import _lar  # noqa: F401

try:
    from brain.hippocampus import Hippocampus
except ImportError:
    raise ImportError(
        "brain.hippocampus not found. Ensure the Lár engine is installed and "
        "SNATH_AVIATION_LARJEPA points to the lar_jepa directory."
    )

def check():
    try:
        print("Connecting to ChromaDB via Hippocampus...")
        h = Hippocampus()
        
        # Query ChromaDB for 'aviation'
        print("Querying for 'aviation' memories...")
        results = h.recall(query="aviation", max_memories=5)
        
        print("\n" + "="*50)
        print("CHROMADB RECALL RESULTS:")
        print("="*50)
        print(results)
        print("="*50)
        
    except Exception as e:
        print(f"Failed to query ChromaDB: {e}")

if __name__ == "__main__":
    check()
