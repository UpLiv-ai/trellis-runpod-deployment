import os
import sys
import json
import torch
import base64
import logging
import numpy as np
from PIL import Image
from io import BytesIO

# Add TRELLIS to Python path
trellis_path = "/trellis-stable-projectorz"
if os.path.exists(trellis_path) and trellis_path not in sys.path:
    sys.path.append(trellis_path)

# Now import TRELLIS
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis import postprocessing_utils

def init():
    """
    Azure ML calls this function once when the container starts.
    Loads the model into memory.
    """
    global model

    # Load the TRELLIS model from the registered model path
    # In Azure ML, your registered model will be mounted at this location
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "image_to_3d_model")
    
    if os.path.exists(model_path):
        model = TrellisImageTo3DPipeline.from_pretrained(model_path)
    else:
        model = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    
    model.cuda()
    logging.info("TRELLIS model initialized successfully")


def run(raw_data):
    """
    Azure ML calls this function to process input requests.
    It takes image(s) as input, runs the 3D generation pipeline, and stores output in Cosmos DB.
    """
    try:
        # Load input images from request
        data = json.loads(raw_data)
        image_list = []

        for img_data in data["images"]:
            img = Image.open(BytesIO(bytes(img_data["image"])))
            image_list.append(img)

        # Generate 3D output
        if(len(image_list) > 1): # Multiple images sent
            outputs = model.run_multi_image(
                image_list,
                seed=1,
                sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
                slat_sampler_params={"steps": 30, "cfg_strength": 3},
            )
        else: # Only one image sent
            image = image_list[0]
            outputs = model.run(
                image,
                seed=1,
                sparse_structure_sampler_params={"steps": 30, "cfg_strength": 7.5},
                slat_sampler_params={"steps": 30, "cfg_strength": 3},
            )

        # Convert the output to GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,    # Reduce triangles by 95%
            texture_size=2048  # Texture size for the GLB
        )

        # Save the GLB file in memory
        glb_buffer = BytesIO()
        glb.export(glb_buffer)
        glb_bytes = glb_buffer.getvalue()

        # Return the GLB file directly as base64-encoded data
        return {
            "status": "Success", 
            "message": "3D model generated successfully",
            "request_id": data.get("request_id", ""),
            "content_type": "model/gltf-binary",
            "glb_data": base64.b64encode(glb_bytes).decode('utf-8')
        }
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return {"status": "Error", "message": str(e)}
