# get packages
import json
import torch
import base64
import io
import warnings
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image


# instantiate model here for warm-starts between requests since we are serverless
from basic_classifier import BasicClassifier
model = None

def load_model() :
    global model
    if model is None :
        #save_dir= Path("model_params")
        best_model_path = "my_best_model.pt" 
        with warnings.catch_warnings() :
            model = BasicClassifier(num_classes = 10)
            warnings.simplefilter("ignore")
            model.load_state_dict(torch.load(best_model_path, map_location = torch.device("cpu")))

    return model

# define preprocessing transform
def convert_to_tensor(image) :
    inference_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

    image_tensor = inference_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def handler(event, context) :
    
    # get the data from the event
    body = json.loads(event.get("body", "{}"))
    image_base64 = body.get("data")
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image_tensor = convert_to_tensor(image) 

    # predict
    model = load_model()
    with torch.no_grad() :
        output = model(image_tensor)
        _, prediction = output.max(1)

    return {
        "statusCode" : 200,
        "body" : json.dumps({"prediction" : int(prediction.item())})
    }
