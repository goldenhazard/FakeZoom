import numpy as np
from sentence_transformers import SentenceTransformer,util

class Answerer:
    def __init__(self, model):
        self.model = model
        self.corpus = []
        self.sentence_embeddings = []
        self.clusters = []

    def find_cluster(self, idx, clusters):
        for i, cluster in enumerate(clusters):
            if idx in cluster:
                return i
        return -1
  
    def update(self, corpus):
        self.corpus = corpus
        self.sentence_embeddings = self.model.encode(corpus)
  
    def cluster(self, min_community_size=1, threshold=0.45):
        clusters = util.community_detection(self.sentence_embeddings, min_community_size=min_community_size, threshold=threshold)
        for i, cluster in enumerate(clusters):
            print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
            for sentence_id in cluster[0:3]:
                print("\t", self.corpus[sentence_id])
            print("\t", "...")
            for sentence_id in cluster[-3:]:
                print("\t", self.corpus[sentence_id])

        self.clusters = clusters

    def find_query_cluster(self, query_sentence, k=3):
        query_embedding = self.model.encode(query_sentence)
        # KNN
        similarity = util.cos_sim(query_embedding, self.sentence_embeddings)
        topk_idx = np.argsort(similarity)[:, -k:][0]
        cluster_count = np.zeros(len(self.clusters))
        for idx in topk_idx:
            cluster_idx = self.find_cluster(idx, self.clusters)
            if cluster_idx != -1:
                cluster_count[cluster_idx] += 1
            
        return cluster_count.argmax()
