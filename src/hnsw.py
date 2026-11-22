"""HNSW graph, written from scratch. A skip list for nearest neighbours."""
import heapq
import math
import random

import numpy as np


class HNSW:
    def __init__(self, dim, M=16, ef_construction=100, ef_search=50, seed=42):
        """Set up an empty HNSW index."""
        self.dim = dim
        self.M = M
        self.M_max = M            # max links per node on layers >= 1
        self.M_max0 = 2 * M       # max links per node on layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self.level_mult = 1.0 / math.log(M)   # "mL" in the paper
        self._rng = random.Random(seed)

        self.vectors = np.empty((1024, dim), dtype=np.float32)  # grows by doubling
        self.count = 0
        self.levels = []          # levels[i] = highest layer node i appears on
        self.links = []           # links[layer][node] -> list of neighbour ids
        self.entry_point = -1     # node where every search starts
        self.max_level = -1       # top layer of the whole graph

    def __len__(self):
        return self.count

    # ------------------------------------------------------------------ helpers
    def _dist(self, ids, q):
        """Cosine distance (1 - dot) from query q to the stored vectors `ids`."""
        return 1.0 - self.vectors[ids] @ q

    def _random_level(self):
        """Draw a node's top layer: floor(-ln(U) * mL). P(level >= l) = M^-l."""
        return int(-math.log(1.0 - self._rng.random()) * self.level_mult)

    def _normalise(self, v):
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _max_links(self, layer):
        return self.M_max0 if layer == 0 else self.M_max

    # ------------------------------------------------- core 1: greedy walk (ef=1)
    def _greedy_closest(self, q, entry, layer):
        """Hill-climb on one layer until no neighbour is closer to the query."""
        current = entry
        current_dist = float(self._dist([current], q)[0])
        improved = True
        while improved:
            improved = False
            neighbours = self.links[layer][current]
            if not neighbours:
                break
            dists = self._dist(neighbours, q)
            best = int(np.argmin(dists))
            if dists[best] < current_dist:
                current, current_dist = neighbours[best], float(dists[best])
                improved = True
        return current

    # ----------------------------------------- core 2: beam search on one layer
    def _search_layer(self, q, entry_points, ef, layer):
        """Best-first beam search of width ef on one layer, closest first."""
        visited = set(entry_points)
        entry_dists = self._dist(entry_points, q).tolist()
        candidates = list(zip(entry_dists, entry_points))
        heapq.heapify(candidates)
        results = [(-d, i) for d, i in candidates]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        layer_links = self.links[layer]
        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst = -results[0][0]
            # Stop when the closest unexpanded candidate is already worse than
            # everything in a full result set: nothing nearer can be reached.
            if len(results) >= ef and d_c > worst:
                break
            new = [n for n in layer_links[c] if n not in visited]
            if not new:
                continue
            visited.update(new)
            dists = self._dist(new, q).tolist()  # one NumPy call for all neighbours
            for n, d in zip(new, dists):
                if len(results) < ef or d < worst:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    if len(results) > ef:
                        heapq.heappop(results)   # drop the current worst
                    worst = -results[0][0]
        return sorted((-nd, i) for nd, i in results)

    # --------------------------------------------- core 3: neighbour selection
    def _select_neighbours(self, candidates, m, keep_pruned=True):
        """Pick up to m links, favouring links that point in different directions."""
        if len(candidates) <= m:
            return [i for _, i in candidates]
        ids = np.fromiter((i for _, i in candidates), dtype=np.int64, count=len(candidates))
        sims_to_q = [1.0 - d for d, _ in candidates]
        V = self.vectors[ids]
        pair_sims = V @ V.T                      # similarity of every candidate pair
        selected, pruned = [], []
        for j in range(len(ids)):
            if len(selected) >= m:
                break
            if not selected or sims_to_q[j] > pair_sims[j, selected].max():
                selected.append(j)               # closer to q than to any kept link
            else:
                pruned.append(j)
        if keep_pruned:
            for j in pruned:
                if len(selected) >= m:
                    break
                selected.append(j)
        return [int(ids[j]) for j in selected]

    # ------------------------------------------------------------------ insert
    def add(self, vector):
        """Insert one vector and return its integer id (ids are 0, 1, 2, ...)."""
        q = self._normalise(vector)
        node = self.count
        if node == len(self.vectors):                       # grow storage
            self.vectors = np.concatenate([self.vectors, np.empty_like(self.vectors)])
        self.vectors[node] = q
        self.count += 1

        # 1. Choose the node's top layer and create empty link lists up to it.
        level = self._random_level()
        self.levels.append(level)
        while len(self.links) <= level:
            self.links.append({})
        for layer in range(level + 1):
            self.links[layer][node] = []

        if self.entry_point == -1:                          # very first node
            self.entry_point, self.max_level = node, level
            return node

        # 2. Above the node's own top layer: just descend greedily to find
        #    a good entry point (no links are made on these layers).
        entry = self.entry_point
        for layer in range(self.max_level, level, -1):
            entry = self._greedy_closest(q, entry, layer)

        # 3. From the node's top layer down to layer 0: beam-search for
        #    candidates, choose links, and add the reverse links.
        entries = [entry]
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, entries, self.ef_construction, layer)
            neighbours = self._select_neighbours(found, self.M)
            self.links[layer][node] = neighbours
            for n in neighbours:                            # bidirectional links
                self._add_link(n, node, layer)
            entries = [i for _, i in found]                 # seed the next layer down

        # 4. A node taller than the whole graph becomes the new entry point.
        if level > self.max_level:
            self.entry_point, self.max_level = node, level
        return node

    def _add_link(self, node, new_neighbour, layer):
        """Add a link, pruning back to the link limit if needed."""
        links = self.links[layer][node]
        links.append(new_neighbour)
        limit = self._max_links(layer)
        if len(links) > limit:
            dists = self._dist(links, self.vectors[node])
            order = np.argsort(dists)
            candidates = [(float(dists[o]), links[o]) for o in order]
            self.links[layer][node] = self._select_neighbours(candidates, limit)

    def add_batch(self, vectors, progress=None):
        """Insert many vectors in order. `progress(i)` is called every 1000 inserts."""
        for i, v in enumerate(vectors):
            self.add(v)
            if progress and (i + 1) % 1000 == 0:
                progress(i + 1)

    # ------------------------------------------------------------------ search
    def search(self, query, k=10, ef=None):
        """Approximate top-k. Returns (ids, cosine_scores), best first."""
        if self.count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        ef = max(ef if ef is not None else self.ef_search, k)
        q = self._normalise(query)
        entry = self.entry_point
        for layer in range(self.max_level, 0, -1):          # express lanes
            entry = self._greedy_closest(q, entry, layer)
        found = self._search_layer(q, [entry], ef, 0)[:k]   # precise local search
        ids = np.array([i for _, i in found], dtype=np.int64)
        scores = np.array([1.0 - d for d, _ in found], dtype=np.float32)
        return ids, scores

    # ------------------------------------------------------------- persistence
    def save(self, path):
        """Write vectors and graph to one .npz file."""
        arrays = {
            "vectors": self.vectors[:self.count],
            "levels": np.array(self.levels, dtype=np.int32),
            "params": np.array([self.dim, self.M, self.ef_construction,
                                self.ef_search, self.seed, self.entry_point,
                                self.max_level], dtype=np.int64),
        }
        for layer, table in enumerate(self.links):
            nodes = np.fromiter(table.keys(), dtype=np.int32, count=len(table))
            lengths = np.fromiter((len(table[n]) for n in nodes.tolist()),
                                  dtype=np.int64, count=len(nodes))
            flat = np.fromiter((x for n in nodes.tolist() for x in table[n]),
                               dtype=np.int32, count=int(lengths.sum()))
            arrays[f"L{layer}_nodes"] = nodes
            arrays[f"L{layer}_offsets"] = np.concatenate([[0], np.cumsum(lengths)])
            arrays[f"L{layer}_flat"] = flat
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        dim, M, ef_c, ef_s, seed, entry, max_level = (int(x) for x in data["params"])
        index = cls(dim, M=M, ef_construction=ef_c, ef_search=ef_s, seed=seed)
        index.vectors = np.ascontiguousarray(data["vectors"])
        index.count = len(index.vectors)
        index.levels = data["levels"].tolist()
        index.entry_point, index.max_level = entry, max_level
        for layer in range(max_level + 1):
            nodes = data[f"L{layer}_nodes"].tolist()
            offsets = data[f"L{layer}_offsets"]
            flat = data[f"L{layer}_flat"].tolist()
            index.links.append({n: flat[offsets[j]:offsets[j + 1]]
                                for j, n in enumerate(nodes)})
        index._rng = random.Random(seed + index.count)  # fresh stream for new inserts
        return index
