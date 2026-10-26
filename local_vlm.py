import io, sys, time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"      # open weights, Apache 2.0, runs fully on your laptop
PROMPT = (
    "You are helping a blind user. Answer in at most 25 words in total, in one or two short "
    "sentences. Start with the closest person or obstacle, and say left, center or right. "
    "Do not give distances in metres and do not say the path is clear."
)


class LocalVLM:
    def __init__(self, model_id=MODEL_ID, max_new_tokens=60):
        print(f"Loading {model_id} (the first run downloads about 4 GB)...")
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16
        ).to("cuda")
        self.model.eval()
        # warm-up so the first real description is not slow
        blank = Image.new("RGB", (320, 240), (128, 128, 128))
        buf = io.BytesIO()
        blank.save(buf, format="JPEG")
        self.describe(buf.getvalue())
        print("Describer ready.")

    def describe(self, jpeg_bytes, prompt=PROMPT):
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        image.thumbnail((640, 640))
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    # one-shot test: grab a single webcam frame and describe it
    import cv2
    vlm = LocalVLM()
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read the webcam.")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    t0 = time.time()
    text = vlm.describe(buf.tobytes())
    print(f"\nDESCRIPTION ({time.time() - t0:.1f} s): {text}")
