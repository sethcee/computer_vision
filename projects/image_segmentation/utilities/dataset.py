# define custom dataset
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import tv_tensors
from PIL import Image
import numpy as np
import os

class MSCOCO(Dataset) :

    def __init__(self, image_directory, annotations_filepath, transform = None) :
        super(MSCOCO, self).__init__()

        # set internval variables
        self.image_directory = image_directory
        self.annotations_filepath = annotations_filepath
        self.transform = transform

        # get person IDs
        self.coco = COCO(annotations_filepath)
        self.ids = self.coco.getImgIds(catIds=[50])


    def __len__(self) :
        return len(self.ids)
    
    def __getitem__(self, idx):

        for attempt in range(len(self)) : 
        
            try :
                # get image id
                image_id = self.ids[idx]

                # get filename from coco meta data
                image_meta_data = self.coco.loadImgs(image_id)[0]
                image_filepath = os.path.join(self.image_directory, image_meta_data['file_name'])

                # load image
                image = Image.open(image_filepath).convert("RGB")
                image = tv_tensors.Image(image)

                # get mask
                annotation_ids = self.coco.getAnnIds(imgIds = image_id, catIds = [50])
                annotations = self.coco.loadAnns(annotation_ids)
                mask = self._make_mask(image_meta_data, annotations)
                mask = tv_tensors.Mask(mask)

                # transform
                if self.transform is not None :

                    image, mask = self.transform(image, mask)

                return image, mask

            except Exception as e :
                idx = (idx + 1) % len(self)


    def _make_mask(self, image_info, annotations) :
        mask = np.zeros((image_info['height'], image_info['width']), dtype = np.uint8)
        for annotation in annotations :
            mask[self.coco.annToMask(annotation) == 1] = 1
        return mask
    