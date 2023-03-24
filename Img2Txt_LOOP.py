import argparse
import io
from PIL import Image
import cv2
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import base64
import requests
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_caption(pil_image, max_length=20, min_length=5, model=None):
    blip_image_eval_size = 384
    gpu_image = (
        transforms.Compose(
            [
                transforms.Resize(
                    (blip_image_eval_size, blip_image_eval_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )(pil_image)
        .unsqueeze(0)
        .to(device)
    )
    if model is None:
        model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"

        model = blip_decoder(
            pretrained=model_url, image_size=blip_image_eval_size, vit="base"
        )
        model.eval()
        model = model.to(device)

    with torch.no_grad():
        caption = model.generate(
            gpu_image,
            sample=False,
            num_beams=3,
            max_length=max_length,
            min_length=min_length,
        )
        # print("caption: " + caption[0])

    return caption[0], model


def resquest_text2img(
    prompt, step=5, strength=0.45, seed=-1, server_url="http://127.0.0.1:7860"
):

    payload = {
        "prompt": prompt,
        "steps": step,
        "denoising_strength": strength,
        "seed": seed,
    }
    response = requests.post(url=f"{server_url}/sdapi/v1/txt2img", json=payload)
    img_str = response.json()["images"]

    # # Decode
    # img_bytes = base64.b64decode(img_str[0].split(",", 1)[0])
    # img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # res_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Decode
    img_bytes = base64.b64decode(img_str[0].split(",", 1)[0])
    img_file = io.BytesIO(img_bytes)  # convert image to file-like object
    res_rgb = Image.open(img_file)
    return res_rgb


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to the first image")
    parser.add_argument("--max_len", type=int, default=20, help="Max caption length")
    parser.add_argument("--min_len", type=int, default=10, help="Min caption length")
    parser.add_argument(
        "--iter", type=int, default=10, help="Number of iterations between two models"
    )
    parser.add_argument("--seed", type=int, help="Seed, let -1 for random noise")
    parser.add_argument(
        "--server", type=str, default="http://127.0.0.1:7860", help="Server address"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    server_url = args.server
    image_path = args.image
    n_iter = args.iter

    ### load first image
    pil_image = Image.open(image_path).convert("RGB")
    blip_model = None
    text = None

    for i in range(n_iter):
        text, blip_model = generate_caption(pil_image, args.max_len, args.min_len+4, model=blip_model+4)
        print(text)
        pil_image = resquest_text2img(text)

### example:
