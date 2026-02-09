import os
import shutil
from typing import List
import requests
from tqdm import tqdm
from enum import Enum
from torchvision import transforms

class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"

def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)

def download_hf_model(model_path: str, model_dir: str = 'models'):
    """
    Download model from updated HuggingFace URLs.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the direct URLs for 1B models
    MODEL_URLS = {
        # Depth models
        "sapiens-depth-1b-torchscript/sapiens_1b_render_people_epoch_88_torchscript.pt2": 
            "https://huggingface.co/facebook/sapiens-depth-1b-torchscript/resolve/main/sapiens_1b_render_people_epoch_88_torchscript.pt2",
        
        # Normal models
        "sapiens-normal-1b-torchscript/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2":
            "https://huggingface.co/facebook/sapiens-normal-1b-torchscript/resolve/main/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
        
        # Segmentation models
        "sapiens-seg-1b-torchscript/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2":
            "https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    }

    # Get filename and create paths
    filename = os.path.basename(model_path)
    temp_path = os.path.join(model_dir, filename)  # .pt2 file
    final_path = os.path.join(model_dir, filename.replace('.pt2', '.pt'))  # .pt file

    # Return if final model already exists
    if os.path.exists(final_path):
        print(f"Model found at {final_path}")
        return final_path

    # Get the correct URL
    if model_path in MODEL_URLS:
        url = MODEL_URLS[model_path]
    else:
        # For other model variants, construct URL
        repo_name = model_path.split("/")[0]
        url = f"https://huggingface.co/facebook/{repo_name}/resolve/main/{filename}"
        print(f"Warning: Using constructed URL for non-1B model variant: {url}")

    print(f"Model not found, downloading from: {url}")
    
    try:
        # Download to temporary file
        download(url, temp_path)
        
        # Verify download
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise Exception("Downloaded file is empty or missing")
        
        # Rename from .pt2 to .pt
        os.rename(temp_path, final_path)
        print(f"Model downloaded and renamed successfully to {final_path}")
        
        return final_path
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Clean up any partial downloads
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(final_path):
            os.remove(final_path)
        raise

def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])

def pose_estimation_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])