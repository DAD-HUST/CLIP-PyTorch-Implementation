from os import device_encoding
import cv2
import torch
from torch.nn import functional as F
import pickle as pkl
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import argparse
from models import CLIPModel
from utils import load_config

IMAGE_EMBEDDINGS_PATH = 'checkpoints/image_embeddings.pkl' 
TEXT_EMBEDDINGS_PATH = 'checkpoints/text_embeddings.pkl' 

def find_matches(model, query_type, query, file_names, device, config, num_results):
    if query_type == "image":
        img = cv2.imread(query)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(float)/255.0
        img = torch.tensor(img).permute(2, 0, 1).float()
        img = img[None, :]

        with torch.no_grad():
            image_features = model.image_encoder(img.to(device))
            image_embeddings = model.image_projection(image_features)
        
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = image_embeddings_n @ text_embeddings_n.T
        
        values, indices = torch.topk(dot_similarity.squeeze(0), 1)
        matches = [text_filenames[idx] for idx in indices[::1]]
        
        print(matches)
        
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(config['text_encoder']['text_tokenizer'])
        encoded_query = tokenizer([query])
        batch = {
            key: torch.tensor(values).to(device_encoding)
            for key, values in encoded_query.items()
        }
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)
        
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_n @ image_embeddings_n.T
        values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
        matches = [image_filenames[idx] for idx in indices[::5]]
        
        _, axes = plt.subplots(3, 3, figsize=(10, 10))
        for match, ax in zip(matches, axes.flatten()):
            image = cv2.imread(f"{CFG.image_path}/{match}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.axis("off")
        
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="model.pth")
    parser.add_argument("--query_type", type=str, default="text")
    parser.add_argument("--query", type=str, default="a photo of a dog")
    parser.add_argument("--num_res", type=int, default=3)
    parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    model = CLIPModel(config).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    find_matches(model, args.query_type, args.query, config, device, num_results=args.num_res)