import logging, os, queue, threading, time, webbrowser
import cv2
from flask import Flask, Response, jsonify, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                                  # model files are found next to this script
MODEL = "yolo26m.pt"
IGNORE = {"tv", "laptop", "keyboard", "mouse", "remote", "cell phone", "book"}
PORT = 8000


def zone(cx, w):
    r = cx / w
    return "left" if r < 0.33 else "right" if r > 0.66 else "ahead"


class Pipeline:
    def __init__(self):
        self.ready, self.error, self.stop = False, "", False
        self.jpeg, self.frame_id = b"", 0
        self.actions = queue.Queue()
        self.events = []
        self.quiet = self.describing = self.speaking = False
        self.hand, self.progress, self.ms = None, 0.0, 0
        self.alert = self.description = ""

    def log(self, kind, text):
        self.events.append({"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text})
        del self.events[:-40]

    def run(self):
        try:
            self._run()
        except BaseException as e:
            self.error = str(e) or e.__class__.__name__
            self.log("error", self.error)

    def _run(self):
        from ultralytics import YOLO
        from speaker import make_speaker
        from announcer import Announcer
        from describer import Describer
        from local_vlm import LocalVLM
        from gesture_control import GestureControl, GestureReader, GatedSpeaker, State

        model = YOLO(MODEL)
        speaker = make_speaker()
        state = State()
        desc = Describer(speaker, LocalVLM())
        ann = Announcer(GatedSpeaker(speaker, state, desc), ignore=IGNORE)
        reader = GestureReader("gesture_recognizer.task")
        control = GestureControl()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Webcam not found. Close other apps that use it, then restart.")

        def do(action, clean):
            if action == "describe":
                ok, buf = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok and desc.request(buf.tobytes()):
                    speaker.say("Describing", interrupt=True)
                    self.log("action", "Describing the scene")
            elif action == "toggle_quiet":
                state.quiet = not state.quiet
                speaker.say("Quiet mode on" if state.quiet else "Sound mode on", interrupt=True)
                self.log("mode", "Quiet mode on" if state.quiet else "Sound mode on")
            elif action == "repeat":
                desc.repeat()

        speaker.say("System ready", interrupt=True)
        self.ready = True
        self.log("system", "Ready")
        seen, seen_desc = None, ""
        while not self.stop:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            h, w = frame.shape[:2]
            clean = frame.copy()                # no boxes drawn: this is what the describer sees
            result = model.track(frame, persist=True, conf=0.5, device=0, verbose=False)[0]
            dets = []
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                label = result.names[int(b.cls)]
                tid = int(b.id) if b.id is not None else None
                dets.append({"id": tid, "label": label, "zone": zone((x1 + x2) / 2, w),
                             "closeness": (x2 - x1) * (y2 - y1) / (w * h)})
                if label not in IGNORE:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 194, 26), 2)
                    cv2.putText(frame, f"{label} #{tid}", (int(x1), max(18, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 194, 26), 2)
            seen = reader.read(clean)
            action, self.progress = control.update(seen, time.time())
            if action:
                self.log("gesture", f"{seen}: {action.replace('_', ' ')}")
                do(action, clean)
            while not self.actions.empty():
                do(self.actions.get_nowait(), clean)
            said = ann.update(dets)
            if said:
                self.alert = said
                self.log("alert", said)
            if desc.last_text and desc.last_text != seen_desc:
                seen_desc = self.description = desc.last_text
                self.log("description", seen_desc)
            self.quiet, self.describing, self.speaking = state.quiet, desc.busy, speaker.busy
            self.hand = seen
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.jpeg = buf.tobytes()
                self.frame_id += 1
            self.ms = int((time.time() - t0) * 1000)


P = Pipeline()
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/video")
def video():
    def gen():
        last = -1
        while True:
            if P.jpeg and P.frame_id != last:
                last = P.frame_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + P.jpeg + b"\r\n"
            time.sleep(0.02)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def get_state():
    return jsonify(ready=P.ready, error=P.error, quiet=P.quiet, describing=P.describing,
                   speaking=P.speaking, hand=P.hand, progress=P.progress, alert=P.alert,
                   description=P.description, ms=P.ms, events=P.events[-8:])


@app.route("/api/<name>", methods=["POST"])
def act(name):
    if name in ("describe", "toggle_quiet", "repeat"):
        P.actions.put(name)
        return jsonify(ok=True)
    if name == "exit":
        P.stop = True
        threading.Timer(0.6, lambda: os._exit(0)).start()
        return jsonify(ok=True)
    return jsonify(ok=False), 404


if __name__ == "__main__":
    threading.Thread(target=P.run, daemon=True).start()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True)
