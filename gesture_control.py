import time

# gesture name -> (action, seconds it must be held)
ACTIONS = {
    "Thumb_Up": ("describe", 0.8),
    "Open_Palm": ("toggle_quiet", 1.2),
}


class State:
    def __init__(self):
        self.quiet = False


class GatedSpeaker:
    """What the announcer talks through. Silent in quiet mode and while a description is running."""

    def __init__(self, speaker, state, describer):
        self.speaker = speaker
        self.state = state
        self.describer = describer

    def say(self, text):
        if self.state.quiet or self.describer.busy:
            return False
        return self.speaker.say(text)


class GestureControl:
    """Turns a stream of gesture labels into actions. A gesture must be HELD, then released to repeat."""

    def __init__(self, cooldown=2.5, grace=0.3):
        self.cooldown = cooldown
        self.grace = grace              # tolerate brief dropouts while the hand is held
        self.current = None
        self.since = 0.0
        self.last_seen = 0.0
        self.fired = False
        self.last_action_time = -1e9

    def update(self, label, now):
        """Returns (action or None, hold progress 0..1)."""
        if label in ACTIONS:
            if label != self.current:
                self.current, self.since, self.fired = label, now, False
            self.last_seen = now
        elif self.current is not None and now - self.last_seen > self.grace:
            self.current, self.fired = None, False
        if self.current is None:
            return None, 0.0
        action, hold = ACTIONS[self.current]
        progress = min(1.0, (now - self.since) / hold)
        if self.fired or now - self.last_action_time < self.cooldown or progress < 1.0:
            return None, progress
        self.fired = True
        self.last_action_time = now
        return action, 1.0


class GestureReader:
    """Reads the current hand gesture from a webcam frame with MediaPipe (open-source, runs locally)."""

    def __init__(self, model_path="gesture_recognizer.task", min_score=0.5):
        import os
        if not os.path.exists(model_path):
            raise SystemExit(f"Missing {model_path}. Download it first (see the PowerShell command).")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.min_score = min_score
        self.t0 = time.time()
        self._last_ts = -1

    def read(self, frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - self.t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self.recognizer.recognize_for_video(image, ts)
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            if top.score >= self.min_score and top.category_name != "None":
                return top.category_name
        return None


if __name__ == "__main__":
    # offline self-test of the hold / release / cooldown logic (no camera, no MediaPipe)
    c = GestureControl()
    fired = []
    t = 0.0
    def feed(label, seconds, fps=20):
        global t
        for _ in range(int(seconds * fps)):
            action, _p = c.update(label, t)
            if action:
                fired.append((round(t, 2), action))
            t += 1 / fps

    feed("Thumb_Up", 0.4); feed(None, 1.0)              # too short: nothing
    feed("Thumb_Up", 2.0); feed(None, 1.0)              # held: ONE describe, not repeated while held
    feed("Open_Palm", 2.0); feed(None, 1.0)             # held: toggle_quiet
    feed("Thumb_Up", 0.5); feed(None, 0.1); feed("Thumb_Up", 0.5); feed(None, 1.0)   # brief dropout is tolerated
    feed("Open_Palm", 0.5); feed("Thumb_Up", 0.5); feed(None, 1.0)                   # switching resets the hold
    print("actions:", fired)

    class Sp:
        def __init__(self): self.said = []
        def say(self, text): self.said.append(text); return True
    class D:
        busy = False
    st, sp, d = State(), Sp(), D()
    g = GatedSpeaker(sp, st, d)
    print("normal:", g.say("person ahead"))
    st.quiet = True
    print("quiet (False):", g.say("person ahead"))
    st.quiet = False; d.busy = True
    print("describing (False):", g.say("person ahead"))
    d.busy = False
    print("normal again:", g.say("person ahead"), sp.said)
