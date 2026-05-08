from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.yolo_person_detector import PERSON_CLASS_ID, load_model
from src.field.pitch_mapper import estimate_field_boundaries, load_polygon, map_to_pitch
from src.team_classification.color_extractor import extract_shirt_color
from src.team_classification.team_assigner import TeamAssigner
from src.tracking.player_stats import PlayerStats
from src.tracking.roster import RosterLabeler
from src.tracking.tracker import track_frame
from src.utils.video import create_writer, frame_limit_reached, open_video
from src.visualization.overlay import (
    MiniPitchPlayer,
    draw_field_boundaries,
    draw_hud,
    draw_mini_pitch,
    draw_player,
    draw_runtime_fps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football AI Analysis POC v1")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source")
    parser.add_argument("--model")
    parser.add_argument("--tracker")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--show", dest="show", action="store_true")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=None)
    parser.add_argument("--save-video", dest="save_video", action="store_true")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.set_defaults(save_video=None)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--team-sample-frames", type=int)
    parser.add_argument("--team-sample-stride", type=int)
    parser.add_argument("--team-min-crops", type=int)
    parser.add_argument("--output-video")
    parser.add_argument("--output-csv")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle) or {}

    for key in (
        "source",
        "model",
        "tracker",
        "conf",
        "iou",
        "imgsz",
        "show",
        "save_video",
        "max_frames",
        "team_sample_frames",
        "team_sample_stride",
        "team_min_crops",
        "output_video",
        "output_csv",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def learn_team_assigner(model: Any, config: Dict[str, Any]) -> TeamAssigner:
    source = str(config["source"])
    sample_frames = int(config.get("team_sample_frames", 150))
    stride = max(1, int(config.get("team_sample_stride", 5)))
    min_crops = int(config.get("team_min_crops", 20))
    features = []

    cap = open_video(source)
    frame_index = 0
    try:
        while cap.isOpened() and frame_index < sample_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            result = model.predict(
                frame,
                conf=float(config.get("conf", 0.25)),
                iou=float(config.get("iou", 0.5)),
                imgsz=int(config.get("imgsz", 960)),
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )[0]

            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().tolist():
                    shirt = extract_shirt_color(frame, box)
                    if shirt is not None:
                        features.append(shirt.feature)
            frame_index += 1
    finally:
        cap.release()

    assigner = TeamAssigner()
    fitted_count = assigner.fit(features)
    if fitted_count < min_crops:
        print(
            f"Warning: only {fitted_count} shirt samples found; "
            "team labels may be unstable or unknown."
        )
    else:
        print(f"Team classifier trained from {fitted_count} shirt samples.")
    return assigner


def write_csv_header(writer: csv.DictWriter) -> None:
    writer.writeheader()


def main() -> None:
    args = parse_args()
    config = load_config(args)

    model = load_model(str(config.get("model", "yolo11n.pt")))
    assigner = learn_team_assigner(model, config)

    source = str(config["source"])
    cap = open_video(source)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    field_polygon = load_polygon(config.get("field_polygon"))
    filter_outside_field = bool(config.get("filter_outside_field", False))
    auto_field_boundaries = bool(config.get("auto_field_boundaries", True))
    draw_boundaries = bool(config.get("draw_field_boundaries", True))
    mini_pitch_enabled = bool(config.get("mini_pitch", True))
    show_confidence_label = bool(config.get("show_confidence_label", False))
    draw_unknown_players = bool(config.get("draw_unknown_players", False))
    draw_bbox = bool(config.get("draw_bbox", False))
    roster = RosterLabeler(
        max_players_per_team=int(config.get("max_players_per_team", 11)),
        release_after_frames=int(config.get("roster_release_frames", 45)),
    )
    player_stats = PlayerStats(
        pitch_length_m=float(config.get("pitch_length_m", 105)),
        pitch_width_m=float(config.get("pitch_width_m", 68)),
        max_speed_kmh=float(config.get("max_player_speed_kmh", 42)),
        display_interval_s=float(config.get("stats_update_interval_s", 1.0)),
    )
    boundary_margin_px = float(config.get("field_boundary_margin_px", 0))

    show = bool(config.get("show", True))
    save_video = bool(config.get("save_video", True))
    max_frames: Optional[int] = config.get("max_frames")
    if max_frames is not None:
        max_frames = int(max_frames)

    writer = None
    if save_video:
        writer = create_writer(str(config["output_video"]), fps, width, height)

    csv_path = Path(str(config["output_csv"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "frame",
        "time_sec",
        "track_id",
        "team",
        "display_id",
        "roster_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "field_x_norm",
        "field_y_norm",
        "in_field",
        "speed_kmh",
        "distance_m",
        "shirt_color_h",
        "shirt_color_s",
        "shirt_color_v",
    ]

    frame_index = 0
    runtime_fps = 0.0
    last_tick = time.perf_counter()
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        write_csv_header(csv_writer)

        try:
            while cap.isOpened() and not frame_limit_reached(frame_index, max_frames):
                ok, frame = cap.read()
                if not ok:
                    break

                result = track_frame(
                    model,
                    frame,
                    tracker=str(config.get("tracker", "botsort.yaml")),
                    conf=float(config.get("conf", 0.25)),
                    iou=float(config.get("iou", 0.5)),
                    imgsz=int(config.get("imgsz", 960)),
                )

                field_boundaries = estimate_field_boundaries(frame) if auto_field_boundaries else None
                team_counts = {"A": 0, "B": 0, "U": 0}
                mini_players = []
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().tolist()
                    ids = result.boxes.id.int().cpu().tolist()
                    confs = result.boxes.conf.cpu().tolist()

                    for xyxy, track_id, confidence in zip(boxes, ids, confs):
                        pitch_point = map_to_pitch(
                            xyxy,
                            width,
                            height,
                            field_polygon,
                            field_boundaries,
                            boundary_margin_px,
                        )
                        if filter_outside_field and not pitch_point.in_field:
                            continue

                        shirt = extract_shirt_color(frame, xyxy)
                        prediction = assigner.predict(track_id, shirt.feature if shirt else None)
                        team = prediction.team if prediction.team in team_counts else "U"
                        if team == "U" and not draw_unknown_players:
                            continue
                        label = roster.label_for(track_id, team, frame_index)
                        motion = player_stats.update(label, pitch_point.x_norm, pitch_point.y_norm, fps)
                        team_counts[team] += 1
                        mini_players.append(
                            MiniPitchPlayer(
                                x_norm=pitch_point.x_norm,
                                y_norm=pitch_point.y_norm,
                                team=team,
                                label=label,
                                in_field=pitch_point.in_field,
                            )
                        )

                        draw_player(
                            frame,
                            xyxy,
                            label,
                            team,
                            float(confidence),
                            show_confidence_label,
                            motion.speed_kmh,
                            motion.distance_m,
                            draw_bbox,
                        )

                        x1, y1, x2, y2 = [float(v) for v in xyxy]
                        h_val, s_val, v_val = shirt.hsv if shirt else ("", "", "")
                        csv_writer.writerow(
                            {
                                "frame": frame_index,
                                "time_sec": f"{frame_index / fps:.3f}",
                                "track_id": track_id,
                                "team": team,
                                "display_id": label,
                                "roster_id": label,
                                "confidence": f"{float(confidence):.4f}",
                                "x1": f"{x1:.2f}",
                                "y1": f"{y1:.2f}",
                                "x2": f"{x2:.2f}",
                                "y2": f"{y2:.2f}",
                                "center_x": f"{(x1 + x2) / 2.0:.2f}",
                                "center_y": f"{(y1 + y2) / 2.0:.2f}",
                                "field_x_norm": f"{pitch_point.x_norm:.4f}",
                                "field_y_norm": f"{pitch_point.y_norm:.4f}",
                                "in_field": int(pitch_point.in_field),
                                "speed_kmh": f"{motion.speed_kmh:.3f}",
                                "distance_m": f"{motion.distance_m:.3f}",
                                "shirt_color_h": f"{h_val:.2f}" if h_val != "" else "",
                                "shirt_color_s": f"{s_val:.2f}" if s_val != "" else "",
                                "shirt_color_v": f"{v_val:.2f}" if v_val != "" else "",
                            }
                        )

                if draw_boundaries and field_boundaries is not None:
                    draw_field_boundaries(frame, field_boundaries)
                draw_hud(frame, frame_index, fps, (team_counts["A"], team_counts["B"], team_counts["U"]))
                if mini_pitch_enabled:
                    draw_mini_pitch(
                        frame,
                        mini_players,
                        width=int(config.get("mini_pitch_width", 300)),
                        height=int(config.get("mini_pitch_height", 185)),
                    )

                now = time.perf_counter()
                elapsed = max(now - last_tick, 1e-6)
                instant_fps = 1.0 / elapsed
                runtime_fps = instant_fps if runtime_fps == 0.0 else (0.9 * runtime_fps + 0.1 * instant_fps)
                last_tick = now
                draw_runtime_fps(frame, runtime_fps)

                if writer is not None:
                    writer.write(frame)

                if show:
                    cv2.imshow("Football AI Analysis POC v1", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames.")
    if save_video:
        print(f"Annotated video: {config['output_video']}")
    print(f"Tracking CSV: {csv_path}")


if __name__ == "__main__":
    main()
