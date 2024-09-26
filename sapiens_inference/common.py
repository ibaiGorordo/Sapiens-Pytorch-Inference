import os
import shutil
from typing import List
import requests
from tqdm import tqdm
from enum import Enum
from huggingface_hub import hf_hub_download, hf_hub_url

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

            # tqdm has many interesting parameters. Feel free to experiment!
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


def download_hf_model(model_name: str, task_type: TaskType, model_dir: str = 'models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    path = model_dir + "/" + model_name
    if os.path.exists(path):
        return path

    print(f"Model {model_name} not found, downloading from Hugging Face Hub...")

    model_version = "_".join(model_name.split("_")[:2])
    repo_id = "facebook/sapiens"
    subdirectory = f"sapiens_lite_host/torchscript/{task_type.value}/checkpoints/{model_version}"

    # hf_hub_download(repo_id=repo_id, filename=model_name, subfolder=subdirectory, local_dir=model_dir)
    url = hf_hub_url(repo_id=repo_id, filename=model_name, subfolder=subdirectory)
    download(url, path)
    print("Model downloaded successfully to", path)

    return path


def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               transforms.Lambda(lambda x: x.unsqueeze(0))
                               ])

def pose_estimation_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               ])
