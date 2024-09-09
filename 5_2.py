import pickle
import faiss
import numpy as np

# Load document embeddings
with open('document_embeddings.pkl', 'rb') as f:
    document_embeddings = pickle.load(f)

# Convert document embeddings to numpy array
document_embeddings_np = np.array(document_embeddings)

# Load claim embeddings
with open('claim_embeddings.pkl', 'rb') as f:
    claim_embeddings = pickle.load(f)

claim_embeddings_np = np.array(claim_embeddings)


ground_truth = [
    [34], 
    [12],
    [45],  
]

# Create FAISS index for document embeddings (L2 distance)
index = faiss.IndexFlatL2(document_embeddings_np.shape[1])
index.add(document_embeddings_np)

# Function to compute MRR and MAP
def evaluate_retrieval(claim_embeddings, ground_truth, index, k=10):
    mrr_total = 0
    map_total = 0
    num_claims = len(claim_embeddings)

    for idx, claim in enumerate(claim_embeddings):
   
        D, I = index.search(np.array([claim]), k=k)
        retrieved_docs = I[0]

        # Calculate MRR
        rr = 0
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in ground_truth[idx]:
                rr = 1 / rank
                break
        mrr_total += rr

        # Calculate MAP
        num_relevant = 0
        precision_at_ranks = []
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in ground_truth[idx]:
                num_relevant += 1
                precision_at_ranks.append(num_relevant / rank)

        if len(ground_truth[idx]) > 0:
            ap = np.mean(precision_at_ranks) if precision_at_ranks else 0
        else:
            ap = 0  

        map_total += ap

    # Compute final MRR and MAP
    mrr = mrr_total / num_claims
    mean_ap = map_total / num_claims

    return mrr, mean_ap

# Evaluate the system
mrr, map_score = evaluate_retrieval(claim_embeddings_np, ground_truth, index, k=10)

print(f"MRR: {mrr:.4f}")
print(f"MAP: {map_score:.4f}")
