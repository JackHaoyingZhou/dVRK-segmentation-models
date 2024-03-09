import os
import sys
from glob import glob
dynamic_path = os.path.abspath(__file__+"/../../")
# print(dynamic_path)
sys.path.append(dynamic_path)
from typing import List
import numpy as np
import cv2
import argparse


def sort_file_list(file_list: List[str]) -> List[str]:
    '''
    Sort a list of files and return the sorted list.
    :param file_list: list of files to sort
    '''
    new_list = sorted(file_list, key=lambda x: x.split('/')[-1].split('.')[0])
    return new_list


def color_to_binary(src_path: str, dest_folder: str, new_name: str) -> None:
    '''
    Convert color image to binary image
    :param src_path: source image file path
    :param dest_folder: destination folder path
    :param new_name: destination image file name
    '''
    # Copy and Rename the copied file
    new_path = os.path.join(dest_folder, new_name)
    color_segment = cv2.imread(src_path)  # bgr
    # Define color range
    lower_bound = np.array([0, 80, 0])
    upper_bound = np.array([0, 255, 0])
    binary_segment = cv2.inRange(color_segment, lower_bound, upper_bound)
    cv2.imwrite(new_path, binary_segment)


class BinarySegment:
    def __init__(self, src_folder: str):
        '''
        :param src_folders: source folders path
        '''
        self.src_folder = src_folder
        self.subfolder_name = 'segmented'
        self.outfolder_name= 'binary_segmented'

        assert os.path.exists(os.path.join(self.src_folder, self.subfolder_name)), 'Segmented Image folder does not exist!'

        # Create folders if necessary
        out_path = os.path.join(self.src_folder, self.outfolder_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print("Created folder " + out_path)
        self.all_img_files = self.get_file_list(self.src_folder, self.subfolder_name)

    @staticmethod
    def get_file_list(folder_path: str, subfolder_name: str) -> List[str]:
        '''
        Get all names of the files
        :param folder_path: source folder path list
        :param subfolder_name: subfolder name
        '''
        file_list = []
        subfolder_path = os.path.join(folder_path, subfolder_name)
        img_list = sort_file_list(glob(os.path.join(subfolder_path, '*.png')))
        file_list.extend(img_list)
        return file_list

    def main(self):
        out_path = os.path.join(self.src_folder, self.outfolder_name)
        num_img = len(self.all_img_files)

        for i_img in range(num_img):
            new_name = self.all_img_files[i_img].split('/')[-1]
            # print(new_name)
            color_to_binary(self.all_img_files[i_img], out_path, new_name)

        print('Binary Segment Convert Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Color to Binary Segmentation')
    parser.add_argument('-i', '--input_dir', required=False, default='../monai_data/AMBF_DATASETS_NEW/train',
                        help='The folder of your data')
    args = parser.parse_args()
    data_path = os.path.abspath(args.input_dir)

    # data_path = os.path.join(dynamic_path, "monai_data", "AMBF_DATASETS_NEW")

    binary_seg = BinarySegment(data_path)

    binary_seg.main()

    print('All Done!')

