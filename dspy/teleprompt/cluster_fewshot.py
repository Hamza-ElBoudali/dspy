import dsp
import random
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .teleprompt import Teleprompter
# from .utils import optimal_k_silhouette


class ClusterFewShot(Teleprompter):
    def __init__(self, num_labeled_demos=16, vectorizer: dsp.BaseSentenceVectorizer = None, clustering_algorithm='kmeans', max_k=10):
        self.num_labeled_demos = num_labeled_demos
        self.vectorizer = vectorizer or dsp.FastEmbedVectorizer()
        self.clustering_algorithm = clustering_algorithm
        self.max_k = max_k

    def optimal_k_silhouette(self, embeddings, max_k):
        silhouette_scores = []
        for k in range(2, min(max_k, len(embeddings)) + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_k = np.argmax(silhouette_scores) + 2

        return optimal_k

    def compile(self, student, *, trainset):
        self.student = student.reset_copy()

        embeddings = self.vectorizer([example.question + " " + example.answer for example in trainset])

        if self.clustering_algorithm == 'kmeans':
            k = self.optimal_k_silhouette(embeddings, self.max_k)
            kmeans = KMeans(n_clusters=k, random_state=0)
            cluster_labels = kmeans.fit_predict(embeddings)
            cluster_assignments = {cluster_id: [trainset[i] for i in np.where(cluster_labels == cluster_id)[0]] for cluster_id in cluster_labels}

            base_samples_per_cluster = self.num_labeled_demos // k
            remainder = self.num_labeled_demos % k  # This is the extra samples we need to distribute

            samples_per_cluster = [base_samples_per_cluster] * k
            eligible_clusters = [cluster_id for cluster_id in range(k) if np.sum(cluster_labels == cluster_id) > base_samples_per_cluster]
            additional_samples_clusters = random.sample(eligible_clusters, min(remainder, len(eligible_clusters)))
            for cluster_id in additional_samples_clusters:
                samples_per_cluster[cluster_id] += 1

            sampled_indices = []
            for cluster_id in range(k):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                sample_size = min(samples_per_cluster[cluster_id], len(cluster_indices))
                sampled_cluster_indices = random.sample(list(cluster_indices), sample_size)
                sampled_indices.extend(sampled_cluster_indices)

            if len(sampled_indices) < self.num_labeled_demos:
                remaining_needed = self.num_labeled_demos - len(sampled_indices)
                all_indices = set(range(len(embeddings)))
                unselected_indices = list(all_indices - set(sampled_indices))
                
                additional_indices = random.sample(unselected_indices, remaining_needed)
                sampled_indices.extend(additional_indices)
        
        elif self.clustering_algorithm == 'hdbscan':
            raise NotImplementedError("HDBSCAN clustering is not yet implemented.")

        for predictor in self.student.predictors():
            predictor.demos = [trainset[i] for i in sampled_indices]
            predictor.cluster_assignments = cluster_assignments
        
        return self.student