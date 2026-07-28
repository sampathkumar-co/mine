from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.cluster import KMeans

Problem = dict[str, object]
Solution = list[int]


def _data(problem: Problem) -> tuple[np.ndarray, int, int]:
    data = np.asarray(problem["X"], dtype=np.float64)
    clusters = int(problem["k"])
    if data.ndim != 2 or clusters <= 0 or data.shape[0] % clusters != 0:
        raise ValueError("invalid k-means problem dimensions")
    block = data.shape[0] // clusters
    return data, clusters, block


def _block_centers(data: np.ndarray, clusters: int, block: int) -> np.ndarray:
    return np.ascontiguousarray(data.reshape(clusters, block, data.shape[1]).mean(axis=1))


def block_elkan(problem: Problem) -> Solution:
    data, clusters, block = _data(problem)
    centers = _block_centers(data, clusters, block)
    model = KMeans(
        n_clusters=clusters,
        init=centers,
        n_init=1,
        max_iter=20,
        tol=1e-3,
        algorithm="elkan",
        random_state=0,
    )
    return model.fit_predict(data).astype(np.int64, copy=False).tolist()


def block_lloyd(problem: Problem) -> Solution:
    data, clusters, block = _data(problem)
    centers = _block_centers(data, clusters, block)
    model = KMeans(
        n_clusters=clusters,
        init=centers,
        n_init=1,
        max_iter=12,
        tol=1e-3,
        algorithm="lloyd",
        random_state=0,
    )
    return model.fit_predict(data).astype(np.int64, copy=False).tolist()


def _assign(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(data * data, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * (data @ centers.T)
    )
    return np.argmin(distances, axis=1).astype(np.int64, copy=False)


def block_numpy(problem: Problem) -> Solution:
    data, clusters, block = _data(problem)
    centers = _block_centers(data, clusters, block)
    scale = max(float(np.sum(centers * centers)), 1.0)
    labels = np.repeat(np.arange(clusters, dtype=np.int64), block)
    for _ in range(8):
        labels = _assign(data, centers)
        counts = np.bincount(labels, minlength=clusters).astype(np.float64)
        updated = centers.copy()
        nonempty = counts > 0.0
        for column in range(data.shape[1]):
            sums = np.bincount(labels, weights=data[:, column], minlength=clusters)
            updated[nonempty, column] = sums[nonempty] / counts[nonempty]
        shift = float(np.sum((updated - centers) ** 2))
        centers = updated
        if shift <= 1e-4 * scale:
            break
        scale = max(float(np.sum(centers * centers)), 1.0)
    return _assign(data, centers).tolist()


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "block_elkan": block_elkan,
    "block_lloyd": block_lloyd,
    "block_numpy": block_numpy,
}
