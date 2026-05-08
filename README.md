# Football Match AI Analysis POC

POC version 1 for football match video analysis.

This project detects people in a football video, tracks them with BoT-SORT, separates Team A and Team B by uniform color, and renders an annotated video with player labels, speed, distance, field boundaries, FPS, and a mini pitch map.

Ball detection is not included yet.

## Current Features

- Player/person detection with YOLO.
- Player tracking with BoT-SORT.
- Automatic Team A / Team B classification from shirt color.
- Roster-style labels such as `A1`, `A2`, `B1`, `B2`.
- Maximum 11 visible roster slots per team.
- Reuses roster numbers when a track disappears.
- Rule-based upper/lower field boundary detection.
- Ignores detections outside the detected field boundary.
- Mini pitch in the bottom-right.
- Runtime FPS in the bottom-left.
- Player speed and distance display, updated every 1 second.
- Annotated MP4 output and CSV tracking export.

## Project Structure

```text
F:\football
|-- config.yaml
|-- requirements.txt
|-- src
|   |-- main.py
|   |-- detection
|   |-- field
|   |-- team_classification
|   |-- tracking
|   |-- utils
|   `-- visualization
|-- data
|   `-- raw_videos
`-- outputs
    |-- videos
    `-- csv
```

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The default model is `yolo11n.pt`. If it is not already present, Ultralytics may download it on first run.

## Run

Default run:

```powershell
python src/main.py
```

Quick smoke test:

```powershell
python src/main.py --max-frames 100 --no-show
```

Use another video:

```powershell
python src/main.py --source data/raw_videos/Olympique_Lyonnais.mp4
```

## Outputs

Default output video:

```text
outputs/videos/liverpool_poc_v1.mp4
```

Default CSV:

```text
outputs/csv/liverpool_poc_v1_tracks.csv
```

CSV includes:

- frame/time
- raw tracker ID
- team
- roster label
- bounding box
- normalized field position
- speed in km/h
- distance in meters
- shirt color values

## Important Config

Edit `config.yaml`.

```yaml
source: data/raw_videos/Liverpool.mp4
model: yolo11n.pt
tracker: botsort.yaml
conf: 0.25
imgsz: 960

filter_outside_field: true
auto_field_boundaries: true
draw_field_boundaries: true

max_players_per_team: 11
roster_release_frames: 45

stats_update_interval_s: 1.0
pitch_length_m: 105
pitch_width_m: 68
```

## Current Limitations

- This POC uses pretrained person detection, not a football-specific detector.
- Goalkeeper, referee, substitute, and ball are not separated yet.
- Team classification is color-based, so similar uniforms or lighting changes can cause mistakes.
- Speed and distance are approximate because there is no true homography calibration yet.
- Field boundary detection is rule-based from green pixels and may need tuning for different videos.
- Roster labels are temporary POC labels, not real jersey numbers.

## Next Steps

- Add football-specific YOLO model classes: player, goalkeeper, referee, ball.
- Add manual team color calibration.
- Add pitch homography for accurate speed/distance.
- Add ball detection and possession logic.
- Add export reports and dashboard.
