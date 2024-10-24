__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import argparse
import os
import PIL
import torchvision
import numpy
import cv2
import pandas

from albumentations import PadIfNeeded

from tqdm import tqdm

from animaloc.data import ImageToPatches, PatchesBuffer, save_batch_images

parser = argparse.ArgumentParser(prog='patcher', description='Cut images into patches')

parser.add_argument('root', type=str,
    help='path to the images directory (str)')
parser.add_argument('height', type=int,
    help='height of the patches, in pixels (int)')
parser.add_argument('width', type=int,
    help='width of the patches, in pixels (int)')
parser.add_argument('overlap', type=int,
    help='overlap between patches, in pixels (int)')
parser.add_argument('dest', type=str,
    help='destination path (str)')
parser.add_argument('-csv', type=str,
    help='path to a csv file containing annotations (str). Defaults to None')
parser.add_argument('-min', type=float, default=0.1,
    help='minimum fraction of area for an annotation to be kept (float). Defautls to 0.1')
parser.add_argument('-all', type=bool, default=False,
    help='set to True to save all patches, not only those containing annotations (bool). Defaults to False')

args = parser.parse_args()

def main():

    images_paths = [os.path.join(args.root, p) for p in os.listdir(args.root) if not p.endswith('.csv')]

    if args.csv is not None:
        patches_buffer = PatchesBuffer(args.csv, args.root, (args.height, args.width), overlap=args.overlap, min_visibility=args.min).buffer
        patches_buffer.drop(columns='limits').to_csv(os.path.join(args.dest, 'gt.csv'), index=False)

        if not args.all:
            images_paths = [os.path.join(args.root, x) for x in pandas.read_csv(args.csv)['images'].unique()]

    for img_path in tqdm(images_paths, desc='Exporting patches'):
        pil_img = PIL.Image.open(img_path)
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)

        if args.csv is not None:
            # save all patches
            if args.all:
                patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
                save_batch_images(patches, img_name, args.dest)

            # or only annotated ones
            else:
                padder = PadIfNeeded(
                    args.height, args.width,
                    position = PadIfNeeded.PositionType.TOP_LEFT,
                    border_mode = cv2.BORDER_CONSTANT,
                    value= 0
                    )
                img_ptch_df = patches_buffer[patches_buffer['base_images']==img_name]
                for row in img_ptch_df[['images','limits']].to_numpy().tolist():
                    ptch_name, limits = row[0], row[1]
                    cropped_img = numpy.array(pil_img.crop(limits.get_tuple))
                    padded_img = PIL.Image.fromarray(padder(image = cropped_img)['image'])
                    padded_img.save(os.path.join(args.dest, ptch_name))

        else:
            patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
            save_batch_images(patches, img_name, args.dest)


if __name__ == '__main__':
    main()