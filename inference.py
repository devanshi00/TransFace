import argparse
import cv2
import numpy as np
import torch
import pickle
from backbones import get_model

# Load source_images and source_faces from files
source_faces = []

with open('source_faces.pkl', 'rb') as file:
    source_faces = pickle.load(file)

@torch.no_grad()
def inference(weight, name, images):
    embeddings = []  # Create an empty list to store embeddings

    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()

    for img in images:
        img = np.array(img)  
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        feat = net(img)
        feat = feat[0].numpy()
        embeddings.append(feat)

    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()

    # Use the modified inference function to get embeddings for all images in source_faces
    source_embeddings = inference(args.weight, args.network, source_faces)

    # Now, the 'source_embeddings' list contains embeddings for all images in source_faces
    print("Embeddings for all images:", source_embeddings)

    # Save source_embeddings to a file using pickle
    with open('source_embeddings.pkl', 'wb') as file:
        pickle.dump(source_embeddings, file)

    print("Source embeddings saved to 'source_embeddings.pkl'")


