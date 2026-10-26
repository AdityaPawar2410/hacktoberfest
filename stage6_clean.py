import time
import cv2
from ultralytics import YOLO
from speaker import make_speaker
from announcer import Announcer
from describer import Describer
from local_vlm import LocalVLM
from gesture_control import GestureControl, GestureReader, GatedSpeaker, State

MODEL = "yolo26m.pt"
IGNORE = {"tv", "laptop", "keyboard", "mouse", "remote", "cell phone", "book"}
GESTURE_EVERY = 1          # run the hand model every Nth frame (raise to 2 if the video gets slow)

model = YOLO(MODEL)
speaker = make_speaker()
state = State()
desc = Describer(speaker, LocalVLM())                       # descriptions: always heard, any mode
ann = Announcer(GatedSpeaker(speaker, state, desc), ignore=IGNORE)   # alerts: sound mode only
reader = GestureReader("gesture_recognizer.task")
control = GestureControl()


def zone(cx, w):
    r = cx / w
    return "left" if r < 0.33 else "right" if r > 0.66 else "ahead"


def do(action, frame):
    if action == "describe":
        ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok_jpg and desc.request(buf.tobytes()):
            speaker.say("Describing", interrupt=True)
    elif action == "toggle_quiet":
        state.quiet = not state.quiet
        speaker.say("Quiet mode on" if state.quiet else "Sound mode on", interrupt=True)
    elif action == "repeat":
        desc.repeat()


speaker.say("System ready", interrupt=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam not found. Close other apps using it, or try index 1.")

last_said, said_at = "", 0.0
seen_label, frame_no = None, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    clean = frame.copy()          # untouched copy: no boxes or text, this is what the describer and hand model see
    frame_no += 1
    result = model.track(frame, persist=True, conf=0.5, device=0, verbose=False)[0]

    dets = []
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        label = result.names[int(b.cls)]
        tid = int(b.id) if b.id is not None else None
        dets.append({
            "id": tid,
            "label": label,
            "zone": zone((x1 + x2) / 2, w),
            "closeness": (x2 - x1) * (y2 - y1) / (w * h),
        })
        if label not in IGNORE:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} #{tid}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # gestures
    if frame_no % GESTURE_EVERY == 0:
        seen_label = reader.read(clean)
    action, progress = control.update(seen_label, time.time())
    if action:
        do(action, clean)

    # automatic alerts (silent in quiet mode and while a description runs)
    said = ann.update(dets)
    if said:
        print(said)
        last_said, said_at = said, time.time()

    # overlay
    if time.time() - said_at < 3:
        cv2.putText(frame, last_said, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if state.quiet:
        cv2.putText(frame, "QUIET MODE", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if desc.busy:
        cv2.putText(frame, "describing...", (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    if seen_label:
        cv2.putText(frame, f"hand: {seen_label}", (w - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (w - 260, 40), (w - 260 + int(240 * progress), 52), (255, 255, 255), -1)
    cv2.putText(frame, "thumbs up = describe   open palm = quiet / sound   (keys: d m r q)", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Project Roshni", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("d"):
        do("describe", clean)
    elif key == ord("m"):
        do("toggle_quiet", clean)
    elif key == ord("r"):
        do("repeat", clean)

cap.release()
cv2.destroyAllWindows()
