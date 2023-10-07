import argparse
import cv2
import numpy as np
import torch
import pickle
from backbones import get_model
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import numpy as np
# Determine if an Nvidia GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Define MTCNN module
mtcnn = MTCNN(image_size=120, margin=0, min_face_size=20,
    thresholds=[0.8, 0.9, 0.9], factor=0.709, post_process=True,keep_all=True,device=device)
# Load source_images and source_embeddings from files
source_images = []
source_embeddings = []

with open('source_images.pkl', 'rb') as file:
    source_images = pickle.load(file)

with open('source_embeddings.pkl', 'rb') as file:
    source_embeddings = pickle.load(file)

# ... (Define findCosineDistance and inference functions) ...
@torch.no_grad()
# Function to crop faces and return as a list of PIL Images
def crop_faces(image, boxes):
    face_images = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        face = image.crop((x1, y1, x2, y2))
        face_images.append(face)
    return face_images
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
def inference(weight, name, images):
    embeddings = []  # Create an empty list to store embeddings

    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()

    for img in images:
        img = np.array(img)  # Convert PIL image to NumPy array
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        feat = net(img)
        feat = feat[0].detach().numpy()
        embeddings.append(feat)

    return embeddings

def find_names(image, minConf):
    # Load an image
    image = Image.open(image)  # Replace 'args.image' with your image path
    # Detect faces
    boxes, _ = mtcnn.detect(image)
    # Get the list of cropped face images
    cropped_test_faces = crop_faces(image, boxes)
    test_embeds = inference(args.weight, args.network, cropped_test_faces)
    d = np.zeros((len(source_embeddings), len(test_embeds)))
    for i, s in enumerate(source_embeddings):
        for j, t in enumerate(test_embeds):
            d[i][j] = findCosineDistance(s, t)
    ids = np.argmin(d, axis = 0)
    roll_nos = []
    for j, i in enumerate(ids):
        if 1 - d[i][j] > args.minConf:
            roll_nos.append(source_imgs[i].split("/")[-1].split(".")[0])
        else:
            roll_nos.append("Unknown")
    return ",".join(roll_nos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--minConf', type=float, default=0.5)
    args = parser.parse_args()
    
    names = find_names(args.image, args.minConf)
    print("Roll Numbers:", names)

