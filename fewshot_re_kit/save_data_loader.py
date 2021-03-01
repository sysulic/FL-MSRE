import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from scipy import misc
from tqdm import tqdm
import pickle


class SaveDataset(data.Dataset):
    def __init__(self, name, root):
        self.root = root
        self.name = name.split('_')[0]
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert False

        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        img_path = os.path.join(root, "entity_pair_in_img.json")
        img_info_path = os.path.join(root, "img_info.json")
        self.img_dir = os.path.join(root, 'imgs')
        self.img_data = json.load(open(img_path))
        self.img_info = json.load(open(img_info_path))
        self.differ_img_feature = {}
        self.same_img_feature = {}
        self.save_differ_img_feature_path = os.path.join(root, name + "_differ_scene.pkl")
        self.save_same_img_feature_path = os.path.join(root, name + "same_scene.pkl")

    def __get_cropped_face(self, img, name):
        bbox = self.img_info[img][name]
        bbox = [int(x) for x in bbox]
        full_img = misc.imread(os.path.expanduser(os.path.join(self.img_dir, img)))
        cropped = full_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        aligned = misc.imresize(cropped, (160, 160), interp='bilinear')  # 将人脸图 resize 到 160*160
        prewhitened = prewhiten(aligned)
        prewhitened = np.transpose(prewhitened, (2, 0, 1))

        return prewhitened

    def __get_average_face__(self, img_list, name):
        nrof_imgs = len(img_list)
        average_face = np.zeros((3, 160, 160), dtype=np.float32)
        return average_face
        for i, img in enumerate(img_list):
            cur_face = self.__get_cropped_face(img, name)
            average_face += np.float32(cur_face)
        average_face = average_face / nrof_imgs
        average_face = torch.from_numpy(average_face)
        return average_face

    def __get_weighted_face__(self, img_list, same_scene_list, name):
        nrof_imgs = len(img_list)
        differ_scene_list = [img for img in img_list if img not in same_scene_list]
        average_face = np.zeros((3, 160, 160), dtype=np.float32)
        return average_face
        for i, img in enumerate(same_scene_list):
            cur_face = self.__get_cropped_face(img, name)
            average_face += np.float32(cur_face)

        for i, img in enumerate(differ_scene_list):
            cur_face_i = self.__get_cropped_face(img, name)
            max_similar = -1
            for j, img_2 in enumerate(same_scene_list):
                cur_face_j = self.__get_cropped_face(img_2, name)
                vector_1 = cur_face_i.flatten()
                vector_2 = cur_face_j.flatten()
                similar = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                max_similar = max(similar, max_similar)
            weight = (max_similar + 1) / 2
            average_face += weight * np.float32(cur_face_i)

        average_face = average_face / nrof_imgs
        average_face = torch.from_numpy(average_face)
        return average_face

    def __get_face_in_differ_scene__(self, sbj, obj, sbj_obj):
        sbj_img_list = [img for img_list in self.img_data[sbj].values() for img in img_list]
        for image in [img for s in self.img_data.keys() for o, img_list in self.img_data[s].items() for img
                      in img_list if o == sbj]:
            if image not in sbj_img_list:
                sbj_img_list.append(image)
        try:
            obj_img_list = [img for img_list in self.img_data[obj].values() for img in img_list]
        except:
            obj_img_list = []
        for image in [img for s in self.img_data.keys() for o, img_list in self.img_data[s].items() for img
                      in img_list if o == obj]:
            if image not in obj_img_list:
                obj_img_list.append(image)

        sbj_face = self.__get_weighted_face__(sbj_img_list, self.img_data[sbj][obj], sbj)
        obj_face = self.__get_weighted_face__(obj_img_list, self.img_data[sbj][obj], obj)
        self.differ_img_feature[sbj_obj] = {sbj: sbj_face, obj: obj_face}

    def __get_face_in_same_scene__(self, sbj, obj, sbj_obj):
        sbj_face = self.__get_average_face__(self.img_data[sbj][obj], sbj)
        obj_face = self.__get_average_face__(self.img_data[sbj][obj], obj)
        self.same_img_feature[sbj_obj] = {sbj: sbj_face, obj: obj_face}

    def __saveface__(self):
        for label in self.classes:
            print(label)
            for entry_data in tqdm(self.json_data[label]):
                sbj = entry_data['sbj']
                obj = entry_data['obj']
                sbj_obj = sbj + '-' + obj
                if sbj_obj not in self.differ_img_feature:
                    self.__get_face_in_differ_scene__(sbj, obj, sbj_obj)
                if sbj_obj not in self.same_img_feature:
                    self.__get_face_in_same_scene__(sbj, obj, sbj_obj)

        with open(self.save_differ_img_feature_path, 'wb') as f:
            pickle.dump(self.differ_img_feature, f)

        with open(self.save_same_img_feature_path, 'wb') as f:
            pickle.dump(self.same_img_feature, f)


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

if __name__ == '__main__':
    save_val_face = SaveDataset('val_data', '../data')
    save_val_face.__saveface__()

    save_test_face = SaveDataset('test_data', '../data')
    save_test_face.__saveface__()
