import pickle
import faiss
import numpy as np

with open('document_embeddings.pkl', 'rb') as f:
    document_embeddings = pickle.load(f)

document_embeddings_np = np.array(document_embeddings)

index = faiss.IndexFlatL2(document_embeddings_np.shape[1])
index.add(document_embeddings_np)

with open('claim_embeddings.pkl', 'rb') as f:
    claim_embeddings = pickle.load(f)

claim_embeddings_np = np.array(claim_embeddings)

D, I = index.search(claim_embeddings_np, k=1)

correct_document_ids = [34, 12, 45, 22, 9, 76, 5, 10, 32, 18]  

retrieved_document_ids = I[:, 0]

correct_retrievals = (retrieved_document_ids == correct_document_ids)
accuracy = np.mean(correct_retrievals)

print("Accuracy:", accuracy)
