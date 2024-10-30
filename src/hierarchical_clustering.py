# ----------------------------
#   Hierarchical Clustering
# ----------------------------


class HierarchicalClustering:
    def __init__(self, linkage="single"):
        self.linkage = linkage
        self.clusters = None
        self.dendrogram = []

    def fit(self, X):
        n_samples = X.shape[0]
        self.clusters = {i: [i] for i in range(n_samples)}
        distances = self.compute_initial_distances(X)

        while len(self.clusters) > 1:
            # Find the two closest clusters
            pair = min(distances, key=distances.get)
            c1, c2 = pair
            new_cluster = self.clusters[c1] + self.clusters[c2]
            self.dendrogram.append((c1, c2, distances[pair]))
            # Update clusters
            new_key = max(self.clusters.keys()) + 1
            self.clusters[new_key] = new_cluster
            del self.clusters[c1]
            del self.clusters[c2]
            # Update distances
            distances = self.update_distances(distances, X, c1, c2, new_key)
        return self.dendrogram

    def compute_initial_distances(self, X):
        distances = {}
        n_samples = X.shape[0]
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[(i, j)] = self.compute_distance(X[i], X[j])
        return distances

    def compute_distance(self, x, y):
        return np.linalg.norm(x - y)

    def update_distances(self, distances, X, c1, c2, new_key):
        new_distances = {}
        for (i, j), dist in distances.items():
            if i in [c1, c2] or j in [c1, c2]:
                continue
            new_i = new_key if i in [c1, c2] else i
            new_j = new_key if j in [c1, c2] else j
            if new_i == new_j:
                continue
            if self.linkage == "single":
                new_dist = min(
                    self.compute_distance(X[p], X[q])
                    for p in self.clusters[new_i]
                    for q in self.clusters[new_j]
                )
            elif self.linkage == "complete":
                new_dist = max(
                    self.compute_distance(X[p], X[q])
                    for p in self.clusters[new_i]
                    for q in self.clusters[new_j]
                )
            elif self.linkage == "average":
                total = 0
                count = 0
                for p in self.clusters[new_i]:
                    for q in self.clusters[new_j]:
                        total += self.compute_distance(X[p], X[q])
                        count += 1
                new_dist = total / count
            else:
                raise ValueError("Unsupported linkage method.")
            new_distances[(new_i, new_j)] = new_dist
        return new_distances
