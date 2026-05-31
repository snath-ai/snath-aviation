import json
try:
    from brain.hippocampus import Hippocampus
except ImportError:
    import sys
    sys.path.insert(0, "/Users/aadithya/Desktop/Lar_Main/DMN/lar/src")
    sys.path.insert(0, "/Users/aadithya/Desktop/Lar_Main/DMN/lar")
    from brain.hippocampus import Hippocampus

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
