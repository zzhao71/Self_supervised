from elasticsearch import Elasticsearch
import numpy as np

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

if not es.ping():
    raise ValueError("Connection failed")


claims = ["Claim 1 text", "Claim 2 text", "Claim 3 text"]  
ground_truth = [
    [34],  
    [12, 15],  
    [45] 
]

# Function to compute MRR and MAP
def evaluate_retrieval(claims, ground_truth, es, k=10):
    mrr_total = 0
    map_total = 0
    num_claims = len(claims)

    for idx, claim in enumerate(claims):

        response = es.search(index='documents', body={
            "query": {
                "match": {
                    "text": claim
                }
            }
        })

        retrieved_docs = [int(hit['_id']) for hit in response['hits']['hits'][:k]]

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


    mrr = mrr_total / num_claims
    mean_ap = map_total / num_claims

    return mrr, mean_ap

mrr, map_score = evaluate_retrieval(claims, ground_truth, es, k=10)

print(f"MRR: {mrr:.4f}")
print(f"MAP: {map_score:.4f}")
