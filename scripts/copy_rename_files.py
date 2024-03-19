import copy
import os
import sys
from glob import glob
dynamic_path = os.path.abspath(__file__+"/../../")
print(dynamic_path)
sys.path.append(dynamic_path)
from typing import List, Union
import shutil
import argparse
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

def sort_file_list(file_list: List[str]) -> List[str]:
    '''
    Sort a list of files and return the sorted list.
    :param file_list: list of files to sort
    '''
    new_list = sorted(file_list, key=lambda x: x.split('/')[-1].split('.')[0])
    return new_list


def copy_and_rename(src_path: str, dest_folder: str, new_name: str) -> None:
    '''
    Copy and Rename files from src_path to dest_folder/new_name
    :param src_path: source file path
    :param dest_folder: destination folder path
    :param new_name: destination file name
    '''
    # Copy and Rename the copied file
    new_path = os.path.join(dest_folder, new_name)
    shutil.copy(src_path, new_path)


class FilesRearrange:
    def __init__(self, src_folder: str, des_folder: str, split_flag: bool = True):
        '''
        :param src_folders: source folders path
        :param des_folder: destination folder path
        :param split_flag: whether to split the dataset into train and test sets
        '''
        self.src = src_folder
        assert os.path.exists(self.src), 'Source folder does not exist!'
        self.src_folders = sort_file_list(glob(os.path.join(self.src, '*')))
        self.des_folder = des_folder
        self.subfolder_list = ['rgb', 'segmented']
        self.split_flag = split_flag
        if split_flag:
            self.outfolder_list = ['train', 'valid']
        else:
            self.outfolder_list = ['test']


        # Create folders if necessary
        if not os.path.exists(self.des_folder):
            os.makedirs(self.des_folder)
            print("Created folder " + self.des_folder)

        for out_name in self.outfolder_list:
            out_path = os.path.join(self.des_folder, out_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                print("Created folder " + out_path)
            for subfolder_name in self.subfolder_list:
                subfolder_path = os.path.join(out_path, subfolder_name)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                    print("Created folder " + subfolder_path)

        all_img_files = self.get_file_list(self.src_folders, self.subfolder_list[0])
        all_label_files = self.get_file_list(self.src_folders, self.subfolder_list[1])

        # self.all_img_files = all_img_files
        # self.all_label_files = all_label_files

        if self.split_flag:
            # train 80%, test 20%
            self.train_img_files, self.valid_img_files, self.train_label_files, self.valid_label_files = \
                train_test_split(all_img_files, all_label_files, test_size=0.2, random_state=42)
        else:
            self.train_img_files = copy.deepcopy(all_img_files)
            self.train_label_files = copy.deepcopy(all_label_files)

    @staticmethod
    def get_file_list(folder_list: List[str], subfolder_name: str)->List[str]:
        '''
        Get all names of the files
        :param folder_path: source folders path list
        :param subfolder_name: subfolder name
        '''
        file_list = []
        for folder_path in folder_list:
            subfolder_path = os.path.join(folder_path, subfolder_name)
            img_list = sort_file_list(glob(os.path.join(subfolder_path, '*.png')))
            file_list.extend(img_list)
        return file_list

    def main(self):
        out_train_path = os.path.join(self.des_folder, self.outfolder_list[0])
        assert len(self.train_img_files) == len(self.train_label_files), 'training image and label not equal!'
        num_train = len(self.train_img_files)

        if self.split_flag:
            out_valid_path = os.path.join(self.des_folder, self.outfolder_list[1])
            assert len(self.valid_img_files) == len(self.valid_label_files), 'validation image and label not equal!'
            num_valid = len(self.valid_img_files)

        for i_train in range(num_train):
            copy_and_rename(self.train_img_files[i_train], os.path.join(out_train_path, self.subfolder_list[0]), f'{str(i_train + 1).zfill(6)}.png')
            copy_and_rename(self.train_label_files[i_train], os.path.join(out_train_path, self.subfolder_list[1]), f'{str(i_train + 1).zfill(6)}.png')

        print('Training or Whole Data Set Copy and Rename Done')

        if self.split_flag:
            for i_valid in range(num_valid):
                copy_and_rename(self.valid_img_files[i_valid], os.path.join(out_valid_path, self.subfolder_list[0]), f'{str(i_valid + 1).zfill(6)}.png')
                copy_and_rename(self.valid_label_files[i_valid], os.path.join(out_valid_path, self.subfolder_list[1]), f'{str(i_valid + 1).zfill(6)}.png')
            print('Validation Set Copy and Rename Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy and Rename to Rearrange files')
    parser.add_argument('-i', '--input_dir', required=False, default='../monai_data/02_AMBF_DATASETS',
                        help='The folder of your data')
    parser.add_argument('-o', '--output_dir', required=False, default='../monai_data/AMBF_oldtool',
                        help='The output folder of your reorganized data')
    parser.add_argument('-s', '--split_flag', required=False, default="True", help='The flag of whether split the dataset')

    args = parser.parse_args()
    data_path =  Path(args.input_dir)
    out_folder = Path(args.output_dir)
    flag_split = json.loads(args.split_flag.lower())

    file_reorganizer = FilesRearrange(data_path, out_folder, flag_split)

    file_reorganizer.main()

    # Copy the yaml file
    yaml_name = 'binary_segmap.yaml'
    label_map_path = data_path / yaml_name 

    assert os.path.exists(label_map_path), f'No label config file at {label_map_path}'

    copy_and_rename(label_map_path, out_folder, yaml_name)

    print('All Done')

