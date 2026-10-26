# Project Roshni

*Roshni* means "light". Roshni is an offline camera assistant that tells blind and low-vision users what is around them, out loud, and is controlled with hand gestures. Every model is open source and runs locally on the laptop, so nothing leaves the device.

Built by [team member names] at [hackathon name], 26 October 2026.

## Features

- **Live awareness:** detects and tracks people and objects, and says where they are (left, ahead, right) and how close (far, near, very close).
- **Alerts that don't nag:** an object must be seen for 4 frames, each alert is spoken 3 times and then stays quiet until something changes, people come first, and only the top 2 objects are announced.
- **Scene description on request:** a local vision-language model describes the scene in about 45 words.
- **Hand-gesture control:** thumbs up (held about 1 second) describes the scene. Open palm (held about 1.2 seconds) switches between sound mode and quiet mode.
- **Quiet mode:** stops automatic alerts and cuts off speech instantly. Descriptions still work on request.
- **Dashboard:** a local web page with the live camera view, current mode, last alert and description, gesture progress and an event log, for helpers and judges.
- **Fully offline and open source.**

## How it works

```
Webcam -> YOLO26 detection + tracking -> direction and distance level -> alert logic -> Piper voice
       -> MediaPipe hand gestures ----> describe / quiet mode
       -> clean frame -> Qwen3-VL-2B description -> Piper voice
Flask server -> browser dashboard (live video, mode, log, buttons)
```

| Job | Model / library |
| --- | --- |
| Object detection and tracking | YOLO26 via Ultralytics |
| Scene description | Qwen3-VL-2B-Instruct (Hugging Face Transformers) |
| Hand gestures | MediaPipe Gesture Recognizer |
| Voice | Piper (Windows built-in voice as a fallback) |
| Dashboard | Flask, HTML and CSS |

Check each project's own license. Ultralytics is AGPL-3.0, so this repository is released under AGPL-3.0.

## Requirements

- Windows 10 or 11, Python 3.12
- A webcam
- An NVIDIA GPU is strongly recommended (developed on a laptop RTX 5070 with 8 GB)

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m piper.download_voices en_US-lessac-medium
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task" -OutFile gesture_recognizer.task
```

`requirements-lock.txt` lists the exact versions this was built and tested with. It is for reference, because the GPU build of PyTorch comes from the index above.

The YOLO26 weights and the Qwen3-VL model download automatically on the first run. After that it works with no internet.

## Run

Double-click `start_roshni.bat`, or run `python app_server.py`. The dashboard opens at http://127.0.0.1:8000. A terminal-only version is `python stage6_clean.py`.

## Limitations

Roshni supplements a white cane or guide dog and does not replace them. It can miss glass, stairs and low obstacles. Distance is estimated from object size, not measured. The camera must see the user's hand for gestures to work.

## Next steps

Depth sensing (Depth Anything V2 Small), a chest-worn camera or phone version, and local-language voices.

## License

AGPL-3.0. See `LICENSE`.
