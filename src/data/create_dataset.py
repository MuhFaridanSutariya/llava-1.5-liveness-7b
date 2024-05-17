import base64
import json
import os
import glob
import requests
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
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
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generating Liveness Images Dataset with Captions")
    parser.add_argument("--image_folders", nargs='+', required=True, help="List of directories path to process")
    parser.add_argument("--real_image_labels", default='../client/adaptation_list.txt' ,help="Path to the txt file that contains about real image labels")
    parser.add_argument("--spoof_image_labels", default='../client/test_list.txt', help="Path to the txt file that contains about spoof image labels")


    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the list of directory paths
    directory_paths = args.image_folders
    real_label_list_txt = args.real_image_labels
    attack_label_list_txt = args.spoof_image_labels

    with open(real_label_list_txt, 'r') as file:
        content1 = file.read()
    
    with open(attack_label_list_txt, 'r') as file:
        content2 = file.read()

    load_dotenv(find_dotenv())
    HF_KEY = os.getenv("HF_KEY")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    REAL_IMG_PATH = '../client/rgb/adaptation/'
    ATTACK_IMG_PATH = '../client/rgb/test/'

    login(HF_KEY)

    REAL_IMG_PATH_LIST_AND_LABELS = [(REAL_IMG_PATH + c.splitlines()[0].split(' ')[0] + '.jpg', c.splitlines()[0].split(' ')[2]) for c in content1.splitlines()]
    REAL_IMG_PATHS = [img[0] for img in REAL_IMG_PATH_LIST_AND_LABELS]
    REAL_IMG_LABELS = [label[1] for label in REAL_IMG_PATH_LIST_AND_LABELS]

    real_df = pd.DataFrame({'IMG_PATHS' : REAL_IMG_PATHS, 'LABELS' : REAL_IMG_LABELS})

    ATTACK_IMG_PATH_LIST_AND_LABELS = [(ATTACK_IMG_PATH + c.splitlines()[0].split(' ')[0] + '.jpg', c.splitlines()[0].split(' ')[2]) for c in content2.splitlines()]
    ATTACK_IMG_PATHS = [img[0] for img in ATTACK_IMG_PATH_LIST_AND_LABELS]
    ATTACK_IMG_LABELS = [label[1] for label in ATTACK_IMG_PATH_LIST_AND_LABELS]

    attack_df = pd.DataFrame({'IMG_PATHS' : ATTACK_IMG_PATHS, 'LABELS' : ATTACK_IMG_LABELS})

    combined_df = pd.concat([real_df, attack_df], ignore_index=True)
    combined_df = shuffle(combined_df, random_state=0)
    combined_df = combined_df.reset_index().drop('index', axis=1)[:20000]
    
    train_df = combined_df[:15000]
    eval_df = combined_df[15000:17500]
    test_df = combined_df[17500:]

    train_jpg_files = list(train_df['IMG_PATHS'])
    train_labels = list(train_df['LABELS'])

    eval_jpg_files = list(eval_df['IMG_PATHS'])
    eval_labels = list(eval_df['LABELS'])

    test_jpg_files = list(test_df['IMG_PATHS'])
    test_labels = list(test_df['LABELS'])

    results = []

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

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

    def process_data_split(data_split, image_paths, labels, api_key, prompt):
        results = []
        for idx, path in enumerate(tqdm(image_paths, total=len(image_paths))):
            base64_image = encode_image(path)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
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


            # Save a checkpoint every 500 steps
            if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
                ds = Dataset.from_dict({"messages": results})
                ds = ds.add_column("images", image_paths[:len(results)])
                ds = ds.add_column("labels", labels[:len(results)])
                ds = ds.cast_column("images", Image())
                ds.push_to_hub(f"firqaaa/liveness", split=data_split)
    

    prompt = """
    I ask you to be a liveness image annotator expert to determine if an image "Real" or "Spoof" by analyze it carefully. 
    If an image is a "Spoof" define what kind of attack, is it spoofing attack that used Print(flat), Replay(monitor, laptop), or Mask(paper, crop-paper, silicone)?
    If an image is a "Real" or "Normal" return "No Attack". 
    Whether if an image is "Real" or "Spoof" give an explanation to this.
    Return your response using following format :

    Real/Spoof : 
    Attack Type :
    Explanation :
    """

    # Assuming train_df, eval_df, and test_df are already defined and loaded
    splits = {
        "train": (list(train_df['IMG_PATHS']), list(train_df['LABELS'])),
        "eval": (list(eval_df['IMG_PATHS']), list(eval_df['LABELS'])),
        "test": (list(test_df['IMG_PATHS']), list(test_df['LABELS']))
    }

    # Process each data split
    for split_name, (image_paths, labels) in splits.items():
        process_data_split(split_name, image_paths, labels, OPENAI_API_KEY, prompt)
