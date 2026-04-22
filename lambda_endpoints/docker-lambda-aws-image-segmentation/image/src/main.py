# get packages
import json
import torch
import base64
import io
import warnings
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
import torchvision.transforms.functional as TF

# instantiate model outside of handler so it stays warm during requests since we are serverless
# the reason I am not separating, e.g., the model into its own sub-directory is because Docker flattens
# the directories

from ResidualUNet import ResidualUNet
model = None

def load_model() :
    global model
    if model is None :
        with warnings.catch_warnings() :
            model = ResidualUNet(num_classes = 1)
            best_model_path = "my_best_model.pt"    
            model.load_state_dict(torch.load(best_model_path, map_location = torch.device("cpu")))

    return model

# define preprocessing transform
def preprocess_image(image) :
    inference_transform = v2.Compose([
    v2.Resize(size = (512, 512), antialias = True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize(mean = [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
    ])

    return inference_transform(image)



def handler(event, context) :

    # get data froom event
    body = json.loads(event['body'])
    image_base64 = body.get("data")
    image_bytes = base64.b64decode(image_base64)
    uploaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # predict
    model = load_model()
    model.eval()
    with torch.no_grad() : 

        # get batch
        image_tensor = TF.to_tensor(uploaded_image)
        image_tensor = image_tensor.unsqueeze(0)
        image_transformed = preprocess_image(image_tensor)

        # get output
        image_transformed = image_transformed
        logits = model(image_transformed)

        # predict the logits
        probs = torch.sigmoid(logits).squeeze(0)
        segmentation_mask = (probs > 0.5).bool()
        segmentation_mask  = segmentation_mask.squeeze(0)
        segmentasion_mask_dims = segmentation_mask.shape

        # convert to a png to send back
        buffered = io.BytesIO()
        segmentation_mask = (segmentation_mask.numpy() * 255).astype('uint8')
        segmentation_mask = Image.fromarray(segmentation_mask)
        segmentation_mask.save(buffered, format = "PNG")
        segmentation_mask = base64.b64encode(buffered.getvalue()).decode('utf-8')
        

    # return the boolean map
    return {
        "statusCode" : 200,
        "body" : json.dumps({
            "mask" : segmentation_mask,
            "format" : "png"

        }
        )
    }