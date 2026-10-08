import io
from PIL import Image, UnidentifiedImageError
from ..model import inference


def predict(file_bytes: bytes, model_name=None):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a readable image.")
    return inference.predict_image(img, model_name)


def list_models():
    return {"models": inference.available_models()}
