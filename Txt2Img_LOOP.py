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
from random import randint
from time import time
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

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
    prompt, step=5, strength=0.45, seed=-1, server_url="http://127.0.0.1:7860", negative_prompt = "", sampler_name=""
):

    #TODO: use models/Stable-diffusion/mdjrny-v4.ckpt (move other models to another folder)

    payload = {
        "prompt": prompt,
        "steps": step,
        "denoising_strength": strength,
        "seed": seed,
        "negative_prompt": negative_prompt,
        # "sampler_name": sampler_name
    }
    response = requests.post(url=f"{server_url}/sdapi/v1/txt2img", json=payload)
    img_str = response.json()["images"]

    # Decode
    img_bytes = base64.b64decode(img_str[0].split(",", 1)[0])
    img_file = io.BytesIO(img_bytes)  # convert image to file-like object
    res_rgb = Image.open(img_file)
    return res_rgb


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="A text to start")
    parser.add_argument("--negative_prompt", type=str, help="Things to exclude from photos")
    parser.add_argument("--out_dir", type=str, help="Path to output directory")
    parser.add_argument("--max_len", type=int, default=30, help="Max caption length")
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


def display_pil_img(img_pil):
    img_np = np.array(img_pil)

    # Convert color space from RGB to BGR
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    


    # Display image using cv2.imshow()
    # cv2.imshow("Image", img_cv2)
    # cv2.waitKey(0)
    # winname = "Test"

    # cv2.namedWindow(winname) # Create a named window
    # cv2.moveWindow(winname, 40,30) # Move it to (40,30)
    # cv2.imshow(winname, img_cv2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def text_to_frame(phrase):
    # Create a black frame
    frame = np.zeros((frame_width//2, frame_height, 3), np.uint8)

    # Split the phrase into multiple lines if necessary
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(phrase, font, 1, 2)
    max_text_width = frame_width//2 - 40  # Leave some margin on the sides
    if text_size[0] > max_text_width:
        words = phrase.split()
        lines = []
        line = ''
        for word in words:
            new_line = line + ' ' + word if line else word
            new_line_size, _ = cv2.getTextSize(new_line, font, 1, 2)
            if new_line_size[0] > max_text_width:
                lines.append(line)
                line = word
            else:
                line = new_line
        lines.append(line)
        num_lines = len(lines)
        line_height = text_size[1] + 10  # Add some spacing between lines
        text_y = (frame_height - num_lines * line_height) // 2
        for i, line in enumerate(lines):
            text_size, _ = cv2.getTextSize(line, font, 1, 2)
            text_x = (frame_width//2 - text_size[0]) // 2
            cv2.putText(frame, line, (text_x, text_y + i * line_height), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        text_x = (frame_width//2 - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        cv2.putText(frame, phrase, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    args = parse()

    server_url = args.server
    text = args.text
    n_iter = args.iter
    negative_prompt = args.negative_prompt

    ### load first image
    # pil_image = Image.open(image_path).convert("RGB")
    blip_model = None

    bible_objects = [
        'Ark of the Covenant',
        'Serpent',
        'Burning bush',
        'Tablets of stone',
        'Rainbow',
        'Manna',
        'Ten plagues of Egypt',
        'Staff of Moses',
        'Golden calf',
        'Tabernacle',
        'Shofar',
        'Joshua\'s trumpet',
        'Samson\'s jawbone',
        'King Solomon\'s temple',
        'David\'s sling',
        'Goliath\'s sword',
        'Balaam\'s talking donkey',
        'Jacob\'s ladder',
        'Noah\'s ark',
        'Tower of Babel',
        'Jacob\'s well',
        'Walls of Jericho',
        'Covenant of circumcision',
        'Moses\' bronze serpent',
        'Gideon\'s fleece',
        'David\'s harp',
        'Abraham\'s tent',
        'Solomon\'s throne',
        'Jeremiah\'s potter\'s wheel',
        'Joseph\'s coat of many colors',
        'Isaiah\'s scroll',
        'Sampson\'s hair',
        'Aaron\'s rod',
        'Jonah\'s whale',
        'Daniel\'s lions',
        'Jacob\'s stew',
        'Boaz\'s threshing floor',
        'Elijah\'s chariot of fire',
        'Rebekah\'s water jar',
        'Bathsheba\'s bath',
        'Adam and Eve\'s fig leaves',
        'Jacob\'s birthright',
        'Leah\'s veil',
        'Rachel\'s tomb',
        'Isaac\'s blessing',
        'Abraham\'s sacrifice',
        'Nehemiah\'s wall',
        'Cain\'s mark',
        'Abel\'s offering',
        'Lot\'s wife',
        'Ezekiel\'s wheel',
        'Naaman\'s leprosy',
        'Achan\'s wedge of gold',
        'Pharaoh\'s crown',
        'Reuben\'s sackcloth',
        'Potiphar\'s wife\'s garment',
        'Korah\'s rebellion',
        'Ephod',
        'Urim and Thummim',
        'Jubilee',
        'Passover lamb',
        'Scapegoat',
        'Breastplate of the high priest',
        'Fruit of the Spirit',
        'Grapes of Canaan',
        'Milk and honey',
        'Myrrh',
        'Frankincense',
        'Gold',
        'Silver',
        'Bronze',
        'Purple cloth',
        'Linen',
        'Alabaster jar',
        'Wine',
        'Oil',
        'Bread',
        'Fish',
        'Loaves and fishes',
        'Five smooth stones',
        'Nard',
        'Olive tree',
        'Fig tree',
        'Pomegranate',
        'Cedar tree',
        'Acacia wood',
        'Myrtle tree',
        'Balm of Gilead',
        'Onyx stone',
        'Jasper stone',
        'Sardius stone',
        'Sapphire stone',
        'Emerald stone',
        'Ruby stone',
        'Diamond stone',
        'Pearl',
        'Net',
        'Fishing boat',
        'Anchor'
    ]
    famous_places = [
        "Garden of Eden",
        "Mount Sinai",
        "Jerusalem",
        "Bethlehem",
        "Mount Zion",
        "Mount Moriah",
        "Mount of Olives",
        "Mount Carmel",
        "Mount Nebo",
        "Mount Horeb",
        "Jordan River",
        "Red Sea",
        "Dead Sea",
        "Sea of Galilee",
        "Promised Land",
        "Canaan",
        "Egypt",
        "Babylon",
        "Nineveh",
        "Sodom and Gomorrah",
        "Nazareth",
        "Capernaum",
        "Bethany",
        "Emmaus",
        "Bethsaida",
        "Jericho",
        "Bethel",
        "Hebron",
        "Joppa",
        "Shiloh",
        "Gilgal",
        "Samaria",
        "Golgotha",
        "Calvary",
        "Tabernacle",
        "Temple of Solomon",
        "Herod's Temple",
        "Moses' Tabernacle",
        "Ark of the Covenant",
        "Noah's Ark",
        "Tower of Babel",
        "Jacob's Well",
        "Zion",
        "Valley of Elah",
        "Valley of the Shadow of Death",
        "Valley of Dry Bones",
        "Valley of Achor",
        "Valley of Jehoshaphat",
        "Mount Ararat",
        "Mount Gerizim",
        "Mount Ebal",
        "Mount Tabor",
        "Mount Gilboa",
        "Mount Hermon",
        "Mount Gilead",
        "Mount Ephraim",
        "Mount Paran",
        "Mount Seir",
        "Mount Hor",
        "Mount Gedor",
        "Mount Mizar",
        "Mount Bashan",
        "Mount Zalmon",
        "Mount Perazim",
        "Mount Zaphon",
        "Mount Baal-Hermon",
        "Mount Meron",
        "Mount Ephron",
        "Mount Zemaraim",
        "Mount Carmel Range",
        "Mount Gaash",
        "Mount Halak",
        "Mount Jearim",
        "Mount Jearim",
        "Mount Lasharon",
        "Mount Megiddo",
        "Mount Zalmon",
        "Mount Shapher",
        "Mount Gebal",
        "Mount Paran",
        "Mount Oreb",
        "Mount Peor",
        "Mount Hakkore",
        "Mount Pisgah",
        "Mount Gilboa",
        "Mount Baal-Perazim",
        "Mount Heres",
        "Mount Judi",
        "Mount Hebo",
        "Mount Gilead",
        "Mount Hebron",
        "Mount Ebal",
        "Mount Zalmon",
        "Mount Moriah",
        "Mount Nebo",
        "Mount Carmel",
        "Mount Sinai",
        "Mount Tabor",
        "Mount Zion",
        "Mount of Olives",
        "Mount of Beatitudes",
        "Mount of Transfiguration",
        "Mount of Temptation",
        "Mount of Blessing",
        "Mount of Cursing",
        "Mount of Assembly",
        "Mount of Remembrance",
        "Mount of Atonement",
        "Mount of Sacrifice",
        "Mount of Weeping",
        "Mount of the Lord",
        "Mount of Joy",
        "Mount of Devotion",
        "Mount of Victory"
    ]


    random.shuffle(bible_objects)

    # Define video properties
    frame_width = 1024
    frame_height = 512
    fps = 30
    frame_duration = int(fps * 1)


    for i in range(n_iter):
        if i == 0:
            dir_save = os.path.join("outputs", str(time()))
            os.makedirs(dir_save, exist_ok=True)

            # Create a VideoWriter object to save the video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
            video_output = cv2.VideoWriter(os.path.join(dir_save, 'output.mp4'), fourcc, fps, (frame_width, frame_height))
            frame_img = np.zeros((frame_width//2, frame_height, 3))


        text_added = f"{text}, with {bible_objects[randint(0, len(bible_objects)-1)]} from the holy bible,"
        text_with_style = text_added + " artistic, super realistic, highly detailed"
        text_log = f"{i}\t{text_with_style}"

        ### add text to log file
        print(text_log)
        with open(os.path.join(dir_save, "texts.txt"), "a") as f:
            f.write(text_with_style+"\n")

        ### add frame to video
        frame_txt = text_to_frame(text)
        big_frame = np.concatenate((frame_txt, frame_img), axis=1).astype(np.uint8)

        # ### DEBUG save image
        # path_save = os.path.join(dir_save, f"DEBUG_{i}.png")
        # cv2.imwrite(path_save, big_frame)

        for _ in range(frame_duration):
            video_output.write(big_frame)

        ### generate image
        pil_image = resquest_text2img(text_with_style, negative_prompt=negative_prompt)
        
        img_np = np.array(pil_image)
        frame_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)        
        
        ### save image
        path_save = os.path.join(dir_save, f"{i}.png")
        cv2.imwrite(path_save, frame_img)

        ### add frame to video
        big_frame = np.concatenate((frame_txt, frame_img), axis=1).astype(np.uint8)
        for _ in range(frame_duration):
            video_output.write(big_frame)

        # ### DEBUG save image
        # path_save = os.path.join(dir_save, f"DEBUG_{i}.png")
        # cv2.imwrite(path_save, big_frame)

        
        ### generate new text for next iteration
        text, blip_model = generate_caption(
            pil_image, args.max_len, args.min_len + 4, model=blip_model
        )



                
    # Release the video writer
    video_output.release()

"""Examples: 
    python Txt2Img_LOOP.py --text "A cat and a dog riding a motorbike together like a superstars" --max_len 20 --min_len 20 --iter 20
"""
