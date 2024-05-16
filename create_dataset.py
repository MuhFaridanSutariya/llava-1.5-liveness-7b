import base64
import json
import os
import glob
import requests
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import Dataset, Image
from tqdm.auto import tqdm
from sklearn.utils import shuffle

def encode_image(image_path):
    with open(image_path, "rb") as file:
        image_data = base64.b64encode(file.read()).decode('utf-8')
    return image_data


def format_output(prompt, image_url, response):
    user_content = {
        "role": "user",
        "content": [
            {
                "index": None,
                "text": prompt,
                "type": "text"
            },
            {
                "index": 0,
                "text": None,
                "type": "image"
            }
        ]
    }

    assistant_content = {
        "role": "assistant",
        "content": [
            {
                "index": None,
                "text": response,
                "type": "text"
            }
        ]
    }

    return [user_content, assistant_content]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generating Liveness Images Dataset with Captions")
    parser.add_argument("--image_folders", nargs='+', required=True, help="List of directories path to process")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the list of directory paths
    directory_paths = args.image_folders

    load_dotenv()
    HF_KEY = os.getenv("HF_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    login(HF_KEY)

    jpg_files = []

    for d in directory_paths:
        jpg_files.extend(glob.glob(os.path.join(d, '**', '*.jpg'), recursive=True))

    results = []

    prompt = """
    I ask you to be a liveness image annotator expert to determine if an image "Real" or "Spoof". 
    If an image is a "Spoof" define what kind of attack, is it spoofing attack that used Print(flat), Replay(monitor, laptop), or Mask(paper, crop-paper, silicone)?
    If an image is a "Real" or "Normal" return "No Attack". 
    Whether if an image is "Real" or "Spoof" give an explanation to this.
    Return your response using following format :

    Real/Spoof : 
    Attack Type :
    Explanation :
    """

    jpg_files = shuffle(jpg_files, random_state=20241605)

    for path in tqdm(jpg_files, total=len(jpg_files)):
        base64_image = encode_image(path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        try:
            assistant_response = response.json()["choices"][0]["message"]["content"]
        except:
            assistant_response = ""

        output = format_output(prompt, f"data:image/jpeg;base64,{base64_image}", assistant_response)

        results.append(output)

    ds = Dataset.from_dict({"results": results})

    ds = ds.add_column("images", jpg_files)

    ds = ds.cast_column("images", Image())

    ds.push_to_hub("firqaaa/liveness")
