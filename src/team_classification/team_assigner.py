from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from src.team_classification.color_extractor import ColorFeature


@dataclass(frozen=True)
class TeamPrediction:
    team: str
    distance: float


class TeamAssigner:
    def __init__(self, max_distance: float = 0.45) -> None:
        self._knn: Optional[KNeighborsClassifier] = None
        self._team_by_cluster: Dict[int, str] = {}
        self._track_votes: Dict[int, Counter[str]] = defaultdict(Counter)
        self.max_distance = max_distance

    @property
    def is_ready(self) -> bool:
        return self._knn is not None

    def fit(self, features: Iterable[ColorFeature]) -> int:
        data = np.asarray(list(features), dtype=np.float32)
        if len(data) < 2:
            return 0

        cluster_count = 2 if len(data) >= 2 else 1
        kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=7)
        cluster_ids = kmeans.fit_predict(data)

        centers = kmeans.cluster_centers_
        order = sorted(
            range(cluster_count),
            key=lambda idx: float(np.arctan2(centers[idx][0], centers[idx][1])),
        )
        self._team_by_cluster = {order[0]: "A", order[1]: "B"} if cluster_count == 2 else {order[0]: "A"}
        labels = [self._team_by_cluster[int(cluster_id)] for cluster_id in cluster_ids]

        neighbors = min(5, len(data))
        self._knn = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
        self._knn.fit(data, labels)
        return len(data)

    def predict(self, track_id: int, feature: Optional[ColorFeature]) -> TeamPrediction:
        if self._knn is None or feature is None:
            return TeamPrediction(team=self._stable_team(track_id), distance=float("inf"))

        data = np.asarray([feature], dtype=np.float32)
        distances, _ = self._knn.kneighbors(data, n_neighbors=1)
        distance = float(distances[0][0])
        if distance > self.max_distance:
            team = "U"
        else:
            team = str(self._knn.predict(data)[0])
            self._track_votes[track_id][team] += 1

        return TeamPrediction(team=self._stable_team(track_id, fallback=team), distance=distance)

    def _stable_team(self, track_id: int, fallback: str = "U") -> str:
        votes = self._track_votes.get(track_id)
        if not votes:
            return fallback
        return votes.most_common(1)[0][0]


def display_id(team: str, track_id: int) -> str:
    prefix = team if team in {"A", "B"} else "U"
    return f"{prefix}_{track_id}"
