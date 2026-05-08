from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PlayerMotion:
    speed_kmh: float = 0.0
    distance_m: float = 0.0


@dataclass
class _PlayerState:
    motion: PlayerMotion
    window_distance_m: float = 0.0
    window_elapsed_s: float = 0.0


class PlayerStats:
    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        max_speed_kmh: float = 42.0,
        display_interval_s: float = 1.0,
    ) -> None:
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.max_speed_kmh = max_speed_kmh
        self.display_interval_s = display_interval_s
        self._last_pos: Dict[str, Tuple[float, float]] = {}
        self._state: Dict[str, _PlayerState] = {}

    def update(self, label: str, x_norm: float, y_norm: float, fps: float) -> PlayerMotion:
        x_m = x_norm * self.pitch_length_m
        y_m = y_norm * self.pitch_width_m
        state = self._state.setdefault(label, _PlayerState(motion=PlayerMotion()))
        previous = self._last_pos.get(label)
        frame_dt = 1.0 / max(fps, 1.0)

        if previous is not None:
            dx = x_m - previous[0]
            dy = y_m - previous[1]
            step_m = (dx * dx + dy * dy) ** 0.5
            instant_speed_kmh = step_m / frame_dt * 3.6
            if instant_speed_kmh <= self.max_speed_kmh:
                state.window_distance_m += step_m

            state.window_elapsed_s += frame_dt
            if state.window_elapsed_s >= self.display_interval_s:
                avg_speed_kmh = state.window_distance_m / state.window_elapsed_s * 3.6
                state.motion.speed_kmh = min(avg_speed_kmh, self.max_speed_kmh)
                state.motion.distance_m += state.window_distance_m
                state.window_distance_m = 0.0
                state.window_elapsed_s = 0.0

        self._last_pos[label] = (x_m, y_m)
        return state.motion
