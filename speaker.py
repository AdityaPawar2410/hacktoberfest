import io, queue, threading, time, wave


class BaseSpeaker:
    """Speech state machine. say() drops if busy (or interrupts); stop() cuts speech instantly."""

    def __init__(self):
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._gen = 0
        self._busy = False
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def busy(self):
        return self._busy

    def say(self, text, interrupt=False):
        with self._lock:
            if self._busy:
                if not interrupt:
                    return False
                self._cancel_locked()
            self._busy = True
            self._gen += 1
            gen = self._gen
        self._q.put((text, gen))
        return True

    def stop(self):
        with self._lock:
            self._cancel_locked()

    def _cancel_locked(self):
        self._gen += 1
        self._busy = False
        self._stop_audio()

    def _loop(self):
        self._thread_init()
        while True:
            text, gen = self._q.get()
            if gen != self._gen:
                continue                      # cancelled before it started
            try:
                self._speak(text, lambda: gen != self._gen)
            except Exception as e:
                print("Speech error:", e)
            with self._lock:
                if gen == self._gen:
                    self._busy = False

    # hooks for the real engines
    def _thread_init(self): pass
    def _stop_audio(self): pass
    def _speak(self, text, cancelled): raise NotImplementedError


class PiperSpeaker(BaseSpeaker):
    """Open-source neural voice (Piper), runs fully offline."""

    def __init__(self, model="en_US-lessac-medium.onnx"):
        import os
        if not os.path.exists(model):
            raise FileNotFoundError(f"Voice file {model} not found (run the piper.download_voices command)")
        import numpy as np
        import sounddevice as sd
        from piper import PiperVoice
        self.np, self.sd = np, sd
        self.voice = PiperVoice.load(model)
        self.cache = {}
        self._synth("Ready")                  # warm-up
        super().__init__()

    def _synth(self, text):
        if text in self.cache:
            return self.cache[text]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            rate = wf.getframerate()
            pcm = self.np.frombuffer(wf.readframes(wf.getnframes()), dtype=self.np.int16)
        if len(self.cache) > 300:
            self.cache.clear()
        self.cache[text] = (pcm, rate)
        return pcm, rate

    def _stop_audio(self):
        try:
            self.sd.stop()
        except Exception:
            pass

    def _speak(self, text, cancelled):
        pcm, rate = self._synth(text)
        if cancelled():
            return
        self.sd.play(pcm, rate)
        if cancelled():
            self.sd.stop()
            return
        end = time.time() + len(pcm) / rate + 0.15
        while time.time() < end and not cancelled():
            time.sleep(0.02)
        if cancelled():
            self.sd.stop()


class SapiSpeaker(BaseSpeaker):
    """Fallback: the built-in Windows voice (runs locally, but the voice itself is not open-source)."""

    def __init__(self, rate=1):
        self._rate = rate
        super().__init__()

    def _thread_init(self):
        import pythoncom, win32com.client
        pythoncom.CoInitialize()
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._voice.Rate = self._rate

    def _speak(self, text, cancelled):
        v = self._voice
        v.Speak(text, 1)                      # 1 = asynchronous
        while not cancelled() and not v.WaitUntilDone(50):
            pass
        if cancelled():
            v.Speak("", 3)                    # 3 = asynchronous + purge: cuts the current speech


def make_speaker():
    try:
        sp = PiperSpeaker()
        print("Voice: Piper (open-source)")
        return sp
    except Exception as e:
        print("Piper voice not available:", e)
        print("Voice: falling back to the Windows voice")
        return SapiSpeaker()


if __name__ == "__main__":
    sp = make_speaker()

    def wait():
        while sp.busy:
            time.sleep(0.05)

    print("1. one:", sp.say("One. Testing the voice."))
    time.sleep(0.2)
    print("2. dropped while busy (False):", sp.say("This should not be heard."))
    wait()
    print("3. two:", sp.say("Two. This is the second sentence."))
    wait()
    print("4. long sentence, cut after 1.5 s...")
    sp.say("Three. This is a very long sentence that will be cut off before it finishes, so you should never hear the end of it.")
    time.sleep(1.5)
    sp.stop()
    time.sleep(0.3)
    print("5. interrupt (True):", sp.say("Four. Interrupted and back to normal.", interrupt=True))
    wait()
    print("done")
