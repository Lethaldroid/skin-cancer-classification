import io
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch

from model_utils import AlexNet, preprocess_image, CLASS_NAMES

app = FastAPI(title="Brain Tumor MRI Classifier API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlexNet()
model.load_state_dict(torch.load("../model/model_weights.pth"))
model.to(device)

@app.get("/")
def root():
    return {"status": "ok", "message": "Brain Tumor MRI Classifier API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    try:
        contents = await file.read()
        x = preprocess_image(contents).to(device)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not read/process image: {e}"}
        )

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    prediction = CLASS_NAMES[int(pred_idx)]
    confidence = float(conf)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "classes": CLASS_NAMES,
    }


# For running locally without Docker:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
