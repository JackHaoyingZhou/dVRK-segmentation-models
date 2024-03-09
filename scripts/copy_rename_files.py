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
    def __init__(self, src_folders: List[str], des_folder: str):
        '''
        :param src_folders: source folders path
        :param des_folder: destination folder path
        '''
        self.src_folders = src_folders
        self.des_folder = des_folder
        self.subfolder_list = ['rgb', 'segmented']
        self.outfolder_list = ['train', 'valid']

        # Create folders if necessary
        if not os.path.exists(self.des_folder):
            os.makedirs(self.des_folder)
            print("Created folder " + self.des_folder)

        for out_name in self.outfolder_list:
            out_path = os.path.join(des_folder, out_name)
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

        # train 80%, test 20%
        self.train_img_files, self.valid_img_files, self.train_label_files, self.valid_label_files = \
            train_test_split(all_img_files, all_label_files, test_size=0.2, random_state=42)

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
        out_valid_path = os.path.join(self.des_folder, self.outfolder_list[1])
        assert len(self.train_img_files) == len(self.train_label_files), 'training image and label not equal!'
        assert len(self.valid_img_files) == len(self.valid_label_files), 'validation image and label not equal!'
        num_train = len(self.train_img_files)
        num_valid = len(self.valid_img_files)

        for i_train in range(num_train):
            copy_and_rename(self.train_img_files[i_train], os.path.join(out_train_path, self.subfolder_list[0]), str(i_train + 1).zfill(6))
            copy_and_rename(self.train_label_files[i_train], os.path.join(out_train_path, self.subfolder_list[1]), str(i_train + 1).zfill(6))

        print('Training Set Copy and Rename Done')

        for i_valid in range(num_valid):
            copy_and_rename(self.valid_img_files[i_valid], os.path.join(out_valid_path, self.subfolder_list[0]), str(i_valid + 1).zfill(6))
            copy_and_rename(self.valid_label_files[i_valid], os.path.join(out_valid_path, self.subfolder_list[1]), str(i_valid + 1).zfill(6))

        print('Validation Set Copy and Rename Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy and Rename to Rearrange files')

    parser.add_argument('-i', '--input_dir', required=False, default='../monai_data/AMBF_DATASETS',
                        help='The folder of your data')
    parser.add_argument('-o', '--output_dir', required=False, default='../monai_data/AMBF_DATASETS_NEW',
                        help='The output folder of your reorganized data')
    args = parser.parse_args()
    data_path = os.path.abspath(args.input_dir)
    out_folder = os.path.abspath(args.output_dir)
    # print(data_path)
    # print(out_folder)
    # data_path = os.path.join(dynamic_path, 'monai_data', 'AMBF_DATASETS')
    # out_folder = os.path.join(dynamic_path, 'monai_data', 'AMBF_DATASETS_New')

    subfolder_list = sort_file_list(glob(os.path.join(data_path, '*')))

    file_reorganizer = FilesRearrange(subfolder_list, out_folder)

    file_reorganizer.main()
    print('All Done')

