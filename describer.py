import re, threading, time


def shorten(text, max_words=50):
    """Keep whole sentences while they fit in max_words (always keep the first one)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out, words = [], 0
    for sent in sentences:
        n = len(sent.split())
        if out and words + n > max_words:
            break
        out.append(sent)
        words += n
    return " ".join(out)


class Describer:
    """Runs a describe backend in the background and speaks the result."""

    def __init__(self, speaker, backend, min_gap=3.0):
        self.speaker = speaker
        self.backend = backend            # anything with .describe(jpeg_bytes) -> str
        self.min_gap = min_gap
        self.busy = False
        self.last_text = ""
        self._last_req = -1e9

    def request(self, jpeg_bytes):
        """Start a description in the background. Returns False if one is already running."""
        now = time.time()
        if now - self._last_req < self.min_gap:
            return False
        if self.busy and now - self._last_req < 30:     # safety: a stuck call frees itself
            return False
        self.busy = True
        self._last_req = now
        threading.Thread(target=self._work, args=(jpeg_bytes,), daemon=True).start()
        return True

    def _work(self, jpeg_bytes):
        try:
            text = shorten((self.backend.describe(jpeg_bytes) or "").strip()) or "No description available."
        except Exception as e:
            print("Describer error:", e)
            text = "Description unavailable."
        self.last_text = text
        print("DESCRIPTION:", text)
        self.speaker.say(text, interrupt=True)     # a description is always heard
        self.busy = False

    def repeat(self):
        if self.last_text:
            self.speaker.say(self.last_text, interrupt=True)


if __name__ == "__main__":
    # offline self-test with a fake backend and fake speaker (no model, no camera)
    class FakeBackend:
        def __init__(self, fail=False): self.fail = fail
        def describe(self, jpeg_bytes):
            time.sleep(0.3)
            if self.fail:
                raise RuntimeError("out of memory")
            return "A person stands ahead. A chair is on the left."

    class FakeSpeaker:
        def __init__(self): self.busy_until = 0.0; self.said = []
        def say(self, text, interrupt=False):
            if time.time() < self.busy_until and not interrupt: return False
            self.busy_until = time.time() + 0.5
            self.said.append(text); return True

    long = ("In front of the user is a person wearing a purple shirt, who is seated. "
            "To the left, there is a curtain with a floral pattern, and to the right, a wall with a picture frame and a power outlet.")
    print("0. shorten keeps the first sentence only:", shorten(long))
    sp = FakeSpeaker()
    d = Describer(sp, FakeBackend(), min_gap=1.0)
    sp.say("Describing")
    print("1. request accepted:", d.request(b"x"))
    print("2. second request while busy (False):", d.request(b"x"))
    time.sleep(1.5)
    print("3. spoken:", sp.said)
    d2 = Describer(sp, FakeBackend(fail=True), min_gap=0.0)
    d2.request(b"x")
    time.sleep(1.5)
    print("4. on failure it says:", sp.said[-1])
