import time

BUCKET_NAMES = ["far", "near", "very close"]
UP = [0.08, 0.25]      # closeness needed to ENTER "near" / "very close"
DOWN = [0.05, 0.18]    # closeness below which we LEAVE them (hysteresis)


def next_bucket(cur, closeness):
    b = cur
    while b < 2 and closeness >= UP[b]:
        b += 1
    while b > 0 and closeness < DOWN[b - 1]:
        b -= 1
    return b


class Track:
    def __init__(self, label):
        self.label = label
        self.zone = "ahead"
        self.closeness = 0.0
        self.bucket = 0
        self.seen = 0            # frames seen so far (stability filter)
        self.missed = 0          # frames since last seen
        self.announced = None    # bucket last announced (None = never)
        self.last_spoken = 0.0
        self.count = 0           # how many times the current alert has been spoken


class Announcer:
    def __init__(self, speaker, stable_frames=4, forget_frames=20, global_gap=2.0,
                 label_cooldown=6.0, reminder_gap=10.0, max_items=2,
                 repeats=3, repeat_gap=4.0,
                 priority=("person",), ignore=()):
        self.speaker = speaker
        self.stable_frames = stable_frames
        self.forget_frames = forget_frames
        self.global_gap = global_gap
        self.label_cooldown = label_cooldown
        self.reminder_gap = reminder_gap
        self.repeats = repeats          # each alert is spoken this many times in total
        self.repeat_gap = repeat_gap
        self.max_items = max_items
        self.priority = set(priority)
        self.ignore = set(ignore)
        self.tracks = {}
        self.label_last = {}     # label -> (time, bucket) of last announcement
        self.last_time = -1e9

    def update(self, dets, now=None):
        """dets: list of {"id", "label", "zone", "closeness"}. Returns spoken text or None."""
        now = time.time() if now is None else now

        # 1. update tracks
        seen_ids = set()
        for d in dets:
            tid = d["id"]
            if tid is None or d["label"] in self.ignore:
                continue
            tr = self.tracks.get(tid)
            if tr is None or tr.label != d["label"]:
                tr = self.tracks[tid] = Track(d["label"])
            tr.zone = d["zone"]
            tr.closeness = d["closeness"]
            tr.bucket = next_bucket(tr.bucket, d["closeness"])
            tr.seen += 1
            tr.missed = 0
            seen_ids.add(tid)
        for tid in list(self.tracks):
            if tid not in seen_ids:
                tr = self.tracks[tid]
                tr.missed += 1
                if tr.missed > self.forget_frames:
                    del self.tracks[tid]

        # 2. decide what needs announcing
        candidates = []
        for tid, tr in self.tracks.items():
            if tr.missed or tr.seen < self.stable_frames:
                continue
            min_bucket = 0 if tr.label in self.priority else 1
            if tr.announced is None:
                if tr.bucket < min_bucket:
                    continue
                last = self.label_last.get(tr.label)
                if last and now - last[0] < self.label_cooldown and tr.bucket <= last[1]:
                    tr.announced = tr.bucket        # same thing as before (new ID): stay quiet
                    tr.last_spoken = now
                    tr.count = self.repeats
                    continue
                candidates.append((tid, tr, "new"))
            elif tr.bucket > tr.announced:
                candidates.append((tid, tr, "closer"))
            elif tr.bucket < tr.announced:
                tr.announced = tr.bucket            # moved away: quiet, but re-announce if it returns
            elif tr.count < self.repeats and now - tr.last_spoken >= self.repeat_gap:
                candidates.append((tid, tr, "repeat"))
            elif tr.bucket == 2 and now - tr.last_spoken >= self.reminder_gap:
                candidates.append((tid, tr, "reminder"))
        if not candidates:
            return None

        # 3. rank: urgent first, then priority classes, then closest
        def rank(c):
            _, tr, why = c
            return (tr.bucket == 2 and why in ("new", "closer"), tr.label in self.priority, tr.closeness)
        candidates.sort(key=rank, reverse=True)
        urgent = rank(candidates[0])[0]
        if not urgent and now - self.last_time < self.global_gap:
            return None

        # 4. one sentence per (label, zone, bucket), top few only
        groups = {}
        for tid, tr, why in candidates:
            groups.setdefault((tr.label, tr.zone, tr.bucket), []).append((tr, why))
        chosen = list(groups.items())[:self.max_items]
        parts = []
        for (label, zone, bucket), _ in chosen:
            parts.append(f"{label} {zone}, {BUCKET_NAMES[bucket]}" if bucket else f"{label} {zone}")
        text = ". ".join(parts) + "."

        # 5. speak (drops if still talking) and remember what was said
        if self.speaker.say(text):
            self.last_time = now
            for (label, zone, bucket), members in chosen:
                self.label_last[label] = (now, bucket)
                for tr, why in members:
                    tr.announced = tr.bucket
                    tr.last_spoken = now
                    tr.count = 1 if why in ("new", "closer") else tr.count + 1
            return text
        return None


def _demo():
    class FakeSpeaker:
        def __init__(self):
            self.t = 0.0
            self.busy_until = 0.0
        def say(self, text):
            if self.t < self.busy_until:
                return False
            self.busy_until = self.t + 1.2
            return True

    sp = FakeSpeaker()
    ann = Announcer(sp)
    for f in range(0, 300):
        t = f / 10
        sp.t = t
        dets = []
        if 10 <= f < 60:
            person = 0.03
        elif 60 <= f < 90:
            person = 0.03 + (f - 60) * 0.003
        elif 90 <= f < 120:
            person = 0.12 + (f - 90) * 0.006
        else:
            person = 0.30
        if f >= 10 and not (130 <= f < 136) and f < 250:
            dets.append({"id": 1, "label": "person", "zone": "ahead", "closeness": person})
        if 10 <= f < 250:
            dets.append({"id": 2, "label": "chair", "zone": "left",
                         "closeness": 0.03 if f < 150 else 0.10})
        if f >= 250:
            dets.append({"id": 7, "label": "person", "zone": "ahead", "closeness": 0.30})
        said = ann.update(dets, now=t)
        if said:
            print(f"t={t:5.1f}s  SPEAK: {said}")


if __name__ == "__main__":
    _demo()
