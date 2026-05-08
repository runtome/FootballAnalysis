from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set


class RosterLabeler:
    """Assign compact roster-style labels such as A01 and B11 per track."""

    def __init__(self, max_players_per_team: int = 11, release_after_frames: int = 45) -> None:
        self.max_players_per_team = max_players_per_team
        self.release_after_frames = release_after_frames
        self._label_by_track: Dict[int, str] = {}
        self._team_by_track: Dict[int, str] = {}
        self._number_by_track: Dict[int, int] = {}
        self._last_seen_frame: Dict[int, int] = {}
        self._used_numbers: Dict[str, Set[int]] = defaultdict(set)

    def release_stale(self, current_frame: int) -> None:
        stale_tracks = [
            track_id
            for track_id, last_seen in self._last_seen_frame.items()
            if current_frame - last_seen > self.release_after_frames
        ]
        for track_id in stale_tracks:
            self._release_track(track_id)

    def label_for(self, track_id: int, team: str, frame_index: int) -> str:
        self.release_stale(frame_index)
        if track_id in self._label_by_track:
            current = self._label_by_track[track_id]
            if current.startswith("U") and team in {"A", "B"}:
                self._release_track(track_id)
                return self.label_for(track_id, team, frame_index)
            self._last_seen_frame[track_id] = frame_index
            return self._label_by_track[track_id]

        if team not in {"A", "B"}:
            label = f"U{track_id:02d}" if track_id < 100 else f"U{track_id}"
            self._label_by_track[track_id] = label
            self._team_by_track[track_id] = "U"
            self._last_seen_frame[track_id] = frame_index
            return label

        number = self._first_available_number(team)
        if number is None:
            label = f"U{track_id:02d}" if track_id < 100 else f"U{track_id}"
            self._label_by_track[track_id] = label
            self._team_by_track[track_id] = "U"
            self._last_seen_frame[track_id] = frame_index
            return label

        label = f"{team}{number:02d}"
        self._label_by_track[track_id] = label
        self._team_by_track[track_id] = team
        self._number_by_track[track_id] = number
        self._used_numbers[team].add(number)
        self._last_seen_frame[track_id] = frame_index
        return label

    def _first_available_number(self, team: str) -> Optional[int]:
        used = self._used_numbers[team]
        for number in range(1, self.max_players_per_team + 1):
            if number not in used:
                return number
        return None

    def _release_track(self, track_id: int) -> None:
        team = self._team_by_track.pop(track_id, None)
        number = self._number_by_track.pop(track_id, None)
        if team in {"A", "B"} and number is not None:
            self._used_numbers[team].discard(number)
        self._label_by_track.pop(track_id, None)
        self._last_seen_frame.pop(track_id, None)
