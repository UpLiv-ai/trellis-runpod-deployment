import os
import sys
import json
import torch
import base64
import logging
from PIL import Image
from io import BytesIO

# Import the RunPod SDK
import runpod

# Add the TRELLIS project to the Python path
sys.path.append('/app/trellis-stable-projectorz')
from trellis.modeling_trellis import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# --- Global Scope Model Initialization ---
# This code runs only ONCE when the worker container starts.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set performance-related environment variables
os.environ = 'xformers'
os.environ = 'native'

model = None
try:
    # Define the fixed path to the model inside the container
    model_path = '/app/trellis-stable-projectorz/image_to_3d_model/TRELLIS-image-large'
    
    if os.path.exists(model_path):
        logging.info(f"Loading TRELLIS model from path: {model_path}")
        # Load the pipeline and move it to the GPU
        model = TrellisImageTo3DPipeline.from_pretrained(model_path)
        model.cuda()
        logging.info("TRELLIS model initialized successfully and moved to GPU.")
    else:
        logging.error(f"Model path not found: {model_path}. Model could not be loaded.")
        # In a production scenario, you might want to exit or handle this failure explicitly
        sys.exit(1)

except Exception as e:
    logging.error(f"Error during model initialization: {e}")
    model = None # Ensure model is None if initialization fails
    sys.exit(1)

# --- RunPod Handler Function ---

def handler(job):
    """
    This function is called for each job received by the worker.
    It processes the input images and returns a 3D model in GLB format.
    """
    if model is None:
        return {"error": "Model is not initialized. Worker is in a failed state."}

    job_input = job['input']
    
    try:
        logging.info(f"Processing job {job.get('id', 'N/A')}")

        # 1. Process Input Images
        if 'images' not in job_input or not isinstance(job_input['images'], list):
            return {"error": "Input must contain a JSON array of base64 strings called 'images'."}

        image_list = []
        for img_b64 in job_input['images']:
            image_data = base64.b64decode(img_b64)
            image = Image.open(BytesIO(image_data))
            image_list.append(image)

        if not image_list:
            return {"error": "The 'images' array cannot be empty."}

        logging.info(f"Successfully decoded {len(image_list)} image(s).")

        # 2. Generate 3D Model
        # These parameters are taken directly from the original score.py
        sparse_params = {'steps': 30, 'cfg_strength': 7.5}
        slat_params = {'steps': 30, 'cfg_strength': 3.0}
        seed = job_input.get('seed', 1) # Allow seed to be passed in the request

        logging.info("Running model inference...")
        if len(image_list) > 1:
            outputs = model.run_multi_image(
                image_list,
                seed=seed,
                sparse_structure_sampler_params=sparse_params,
                slat_sampler_params=slat_params
            )
        else:
            outputs = model.run(
                image_list,
                seed=seed,
                sparse_structure_sampler_params=sparse_params,
                slat_sampler_params=slat_params
            )
        logging.info("Model inference completed.")

        # 3. Convert Output to GLB
        logging.info("Converting model output to GLB format...")
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'],
            outputs['mesh'],
            simplify=0.95,
            texture_size=2048
        )
        
        temp_glb_path = "/tmp/output_model.glb"
        os.makedirs("/tmp", exist_ok=True)
        glb.export(temp_glb_path)
        
        with open(temp_glb_path, "rb") as f:
            glb_bytes = f.read()

        # 4. Prepare and Return Response
        glb_data_b64 = base64.b64encode(glb_bytes).decode("utf-8")
        
        response = {
            "status": "success",
            "message": "Model generated successfully.",
            "contentType": "model/gltf-binary",
            "glb_data": glb_data_b64
        }
        logging.info(f"Job {job.get('id', 'N/A')} completed successfully.")
        return response

    except Exception as e:
        logging.error(f"An error occurred during job processing: {e}", exc_info=True)
        return {"error": str(e)}


# --- Start the RunPod Serverless Worker ---
# This line starts the worker and registers the handler function.
runpod.serverless.start({"handler": handler})