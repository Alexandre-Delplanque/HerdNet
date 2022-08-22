import os
import PIL

from typing import Optional, List, Any, Dict

from .register import DATASETS

from .csv import CSVDataset

@DATASETS.register()
class FolderDataset(CSVDataset):
    ''' Class to create a dataset from a folder containing images only, and a CSV file 
    containing annotations.

    This dataset is built on the basis of CSV files containing box coordinates, in 
    [x_min, y_min, x_max, y_max] format, or point coordinates in [x,y] format.

    All images that do not have corresponding annotations in the CSV file are considered as 
    background images. In this case, the dataset will return the image and empty target 
    (i.e. empty lists).

    The type of annotations is automatically detected internally. The only condition 
    is that the file contains at least the keys ['images', 'x_min', 'y_min', 'x_max', 
    'y_max', 'labels'] for the boxes and, ['images', 'x', 'y', 'labels'] for the points. 
    Any additional information (i.e. additional columns) will be associated and returned 
    by the dataset.

    If no data augmentation is specified, the dataset returns the image in PIL format 
    and the targets as lists. If transforms are specified, the conversion to torch.Tensor
    is done internally, no need to specify this. 
    '''

    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        albu_transforms: Optional[list] = None,
        end_transforms: Optional[list] = None
        ) -> None:
        ''' 
        Args:
            csv_file (str): absolute path to the csv file containing 
                annotations
            root_dir (str) : path to the images folder
            albu_transforms (list, optional): an albumentations' transformations 
                list that takes input sample as entry and returns a transformed 
                version. Defaults to None.
            end_transforms (list, optional): list of transformations that takes
                tensor and expected target as input and returns a transformed
                version. These will be applied after albu_transforms. Defaults
                to None.
        '''

        super(FolderDataset, self).__init__(csv_file, root_dir, albu_transforms, end_transforms)

        self.folder_images = [i for i in os.listdir(self.root_dir) 
                                if i.endswith(('.JPG','.jpg','.JPEG','.jpeg'))]
        
        self.anno_keys = self.data.columns

    def _load_image(self, index: int) -> PIL.Image.Image:
        img_name = list(set(self.folder_images))[index]
        img_path = os.path.join(self.root_dir, img_name)

        pil_img = PIL.Image.open(img_path).convert('RGB')
        pil_img.filename = img_name

        return pil_img

    def _load_target(self, index: int) -> Dict[str,List[Any]]:
        img_name = list(set(self.folder_images))[index]
        annotations = self.data[self.data['images'] == img_name]
        anno_keys = list(self.anno_keys)
        anno_keys.remove('images')

        target = {
        'image_id': [index], 
        'image_name': [img_name]
        }

        if len(annotations) > 0:
            for key in anno_keys:
                target.update({key: list(annotations[key])})

                if key == 'annos': 
                    target.update({key: [list(a.get_tuple) for a in annotations[key]]})

        else:
            for key in anno_keys:
                target.update({key: []})
        
        return target

    def __len__(self):
        return len(set(self.folder_images))