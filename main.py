import base64
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import replicate

# Set up the Replicate client with your API key
replicate_client = replicate.Client(api_token="r8_E6041bz4qowDGOtQoVSEvpo6WwHotTu4GTxKD")

# Define FastAPI app
app = FastAPI()

# Helper functions
def image_to_base64(image_path):
    """Converts an image from a local file to base64 string."""
    with open(image_path, "rb") as img_file:
        return "data:image/jpeg;base64," + base64.b64encode(img_file.read()).decode("utf-8")

def is_url(string):
    """Check if the string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Pydantic models for input validation

# Flux 1.1 Pro Ultra Model Input
class FluxUltraInput(BaseModel):
    prompt: str
    image_prompt: str  # URI or base64 for image prompt
    image_prompt_strength: float
    aspect_ratio: str
    safety_tolerance: int
    seed: int
    raw: bool

# Flux Depth Pro Model Input
class FluxDepthProInput(BaseModel):
    prompt: str
    control_image: str  # URI or base64 for control image
    steps: int
    prompt_upsampling: bool
    guidance: float
    safety_tolerance: int

# Flux Redux Dev Model Input
class FluxReduxInput(BaseModel):
    redux_image: str  # URI or base64 for redux image
    aspect_ratio: str
    num_inference_steps: int
    guidance: float
    seed: int

# Helper function to get the URL from replicate output
def extract_url_from_output(output):
    """Extract URL from replicate output if it's a FileOutput object."""
    if isinstance(output, list):
        return output[0].url if hasattr(output[0], "url") else str(output[0])
    elif hasattr(output, "url"):
        return output.url
    return str(output)

# Helper function to convert local file to base64
def convert_local_file_to_base64(file_path):
    """Converts a local image file to base64."""
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"

# Endpoint for Flux 1.1 Pro Ultra Model
@app.post("/flux-ultra")
async def flux_ultra(input_data: FluxUltraInput):
    try:
        # If image_prompt is a local file, convert it to base64
        if not is_url(input_data.image_prompt):
            image_prompt = convert_local_file_to_base64(input_data.image_prompt)
        else:
            image_prompt = input_data.image_prompt

        model_name = "black-forest-labs/flux-1.1-pro-ultra"
        inputs = {
            "prompt": input_data.prompt,
            "image_prompt": image_prompt,
            "image_prompt_strength": input_data.image_prompt_strength,
            "aspect_ratio": input_data.aspect_ratio,
            "safety_tolerance": input_data.safety_tolerance,
            "seed": input_data.seed,
            "raw": input_data.raw,
        }

        # Run the model
        output = replicate_client.run(model_name, input=inputs)

        # Extract the URL from the output
        output_url = extract_url_from_output(output)

        return {"output_url": output_url}

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error: {e}")

# Endpoint for Flux Depth Pro Model
@app.post("/flux-depth-pro")
async def flux_depth_pro(input_data: FluxDepthProInput):
    try:
        # If control_image is a local file, convert it to base64
        if not is_url(input_data.control_image):
            control_image = convert_local_file_to_base64(input_data.control_image)
        else:
            control_image = input_data.control_image

        model_name = "black-forest-labs/flux-depth-pro"
        inputs = {
            "prompt": input_data.prompt,
            "control_image": control_image,
            "steps": input_data.steps,
            "prompt_upsampling": input_data.prompt_upsampling,
            "guidance": input_data.guidance,
            "safety_tolerance": input_data.safety_tolerance,
        }

        # Run the model
        output = replicate_client.run(model_name, input=inputs)

        # Extract the URL from the output
        output_url = extract_url_from_output(output)

        return {"output_url": output_url}

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error: {e}")

# Endpoint for Flux Redux Dev Model
@app.post("/flux-redux-dev")
async def flux_redux_dev(input_data: FluxReduxInput):
    try:
        # If redux_image is a local file, convert it to base64
        if not is_url(input_data.redux_image):
            redux_image = convert_local_file_to_base64(input_data.redux_image)
        else:
            redux_image = input_data.redux_image

        model_name = "black-forest-labs/flux-redux-dev"
        inputs = {
            "redux_image": redux_image,
            "aspect_ratio": input_data.aspect_ratio,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance": input_data.guidance,
            "seed": input_data.seed,
        }

        # Run the model
        output = replicate_client.run(model_name, input=inputs)

        # Extract the URL from the output
        output_url = extract_url_from_output(output)

        return {"generated_image_url": output_url}

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error: {e}")

# Endpoint for uploading an image file (for local image to base64 conversion)
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Convert the uploaded file to base64
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        base64_image_str = f"data:image/{file.filename.split('.')[-1]};base64," + base64_image
        return {"base64_image": base64_image_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
