import base64
import json
import io
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import pipeline
from icecream import ic

MODE = "image-segmentation"
MODEL_TAG = "briaai/RMBG-1.4" 

def decode_image(json_data: dict[str, str]) -> io.BytesIO:
    base64_string = json_data.get("bin")

    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Convert to a PIL image
    image = Image.open(io.BytesIO(image_data))

    return image

def encode_image(image_path: str) -> dict[str, str]:
    # Read the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Encode the image to base64
        base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Create a JSON object
    json_data = {
        "bin": base64_encoded_image
    }
    return json_data

def extract_card(image: Image) -> Image:
    segmentation_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

    # Perform image segmentation
    outputs = segmentation_pipeline(image)
    return outputs

def extract_text(image: Image) -> str:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

    # Prepare the image for the model
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values

    # Generate text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text
"""
image = get_image_from_base64(json_data)

# Now, convert the image to a format compatible with Hugging Face models
# For example, using a feature extractor and a vision model
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

# Prepare the image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# Now you can pass 'inputs' to the model
outputs = model(**inputs)
print(outputs)
"""