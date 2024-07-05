import base64
import io
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel

def get_image_from_base64(json_data):
    # Extract base64 string from JSON
    base64_string = json_data.get("bin")

    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Convert to a PIL image
    image = Image.open(io.BytesIO(image_data))

    return image

# Example usage:

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
