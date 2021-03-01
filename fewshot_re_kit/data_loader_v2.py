import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import pickle
from scipy import misc
from tqdm import tqdm

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, root, use_img=False, differ_scene=False, multi_choose=1):
        self.root = root
        self.name = name.split('_')[0]
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert False, path

        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
        self.use_img = use_img
        self.multi_choose = multi_choose
        if self.use_img:
            img_path = os.path.join(root, "entity_pair_in_img.json")
            img_info_path = os.path.join(root, "img_info.json")
            self.img_dir = os.path.join(root, 'imgs')
            self.img_data = json.load(open(img_path))
            self.img_info = json.load(open(img_info_path))
            self.save_differ_img_feature_path = os.path.join(root, name + "_differ_scene.pkl")
            self.save_same_img_feature_path = os.path.join(root, name + "same_scene.pkl")
            self.differ_scene = differ_scene
            if self.name == "val" or self.name == "test":
                self.differ_img_feature = pickle.load(open(self.save_differ_img_feature_path, 'rb'))
                self.same_img_feature = pickle.load(open(self.save_same_img_feature_path, 'rb'))

    def __getraw__(self, item):
        word, mask = self.encoder.tokenize(item['mask_sentence'])
        return word, mask, item['sbj'], item['obj']

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
                similar = np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
                max_similar = max(similar, max_similar)
            weight = (max_similar + 1) / 2
            average_face += weight * np.float32(cur_face_i)

        average_face = average_face / nrof_imgs
        average_face = torch.from_numpy(average_face)
        return average_face

    def __get_random_face__(self, sbj, obj, write_2_file=False):
        if sbj == "袭人":
            sbj = "花袭人"
        if obj == "袭人":
            obj = "花袭人"
        if self.differ_scene:
            # 不同场景图
            sbj_img_list = [img for img_list in self.img_data[sbj].values() for img in img_list]
            for image in [img for s in self.img_data.keys() for o, img_list in self.img_data[s].items() for img in
                          img_list if o == sbj]:
                if image not in sbj_img_list:
                    sbj_img_list.append(image)
            random_sbj_img = random.sample(sbj_img_list, 1)[0]

            try:
                obj_img_list = [img for img_list in self.img_data[obj].values() for img in img_list]
            except:
                obj_img_list = []
            for image in [img for s in self.img_data.keys() for o, img_list in self.img_data[s].items() for img in
                          img_list if o == obj]:
                if image not in obj_img_list:
                    obj_img_list.append(image)
            if len(obj_img_list) > 1:
                obj_img_list = [img for img in obj_img_list if img != random_sbj_img]
            random_obj_img = random.sample(obj_img_list, 1)[0]

        else:
            random_img = random.sample(self.img_data[sbj][obj], 1)[0]  # 随机抽取一张图片
            random_sbj_img = random_img
            random_obj_img = random_img

        if write_2_file and self.name == 'test':
            with open('./bad_case_input.txt', 'a', encoding='utf-8') as f:
                f.write(str(random_sbj_img))
                f.write('\n')

        sbj_face = self.__get_cropped_face(random_sbj_img, sbj)
        obj_face = self.__get_cropped_face(random_obj_img, obj)
        sbj_face = torch.from_numpy(np.float32(sbj_face))
        obj_face = torch.from_numpy(np.float32(obj_face))
        return sbj_face, obj_face

    def __getface__(self, sbj, obj, write_2_file=False):
        sbj_face, obj_face = self.__get_random_face__(sbj, obj, write_2_file)
        return sbj_face, obj_face

        # if self.name == 'train':
        #     sbj_face, obj_face = self.__get_random_face__(sbj, obj)
        # else:
        #     # 取所有人脸的均值
        #     tri = sbj + '-' + obj
        #     if self.differ_scene:
        #         # 不同场景图
        #         sbj_face = self.differ_img_feature[tri][sbj]
        #         obj_face = self.differ_img_feature[tri][obj]
        #     else:
        #         sbj_face = self.same_img_feature[tri][sbj]
        #         obj_face = self.same_img_feature[tri][obj]
        #
        # return sbj_face, obj_face


    def __additem__(self, d, word, mask):
        d['word'].append(word)
        d['mask'].append(mask)

    def __addface__(self, d, sbj_face, obj_face):
        d['sbj_face'].append(sbj_face)
        d['obj_face'].append(obj_face)

    def __getitem__(self, index):
        support_set = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}
        query_set = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}

        sentence_count = 0
        for cls in self.classes:
            sentence_count += len(self.json_data[cls])

        if index > sentence_count:
            assert False

        qurey_index = 0
        query_class = None
        for cls in self.classes:
            if qurey_index + len(self.json_data[cls]) > index:
                qurey_index = index - qurey_index
                query_class = cls
                break
            else:
                qurey_index += len(self.json_data[cls])
        if qurey_index >= len(self.json_data[query_class]):
            qurey_index = len(self.json_data[query_class]) - 1
        target_classes = random.sample([cls for cls in self.classes if cls != query_class], self.N - 1)
        target_classes.append(query_class)
        q_word, q_mask, q_sbj, q_obj = self.__getraw__(self.json_data[query_class][qurey_index])
        q_word = torch.tensor(q_word).long()
        q_mask = torch.tensor(q_mask).long()
        self.__additem__(query_set, q_word, q_mask)
        if self.use_img:
            sbj_face, obj_face = self.__getface__(q_sbj, q_obj, write_2_file=True)
            self.__addface__(query_set, sbj_face, obj_face)
        query_label = [self.N - 1]

        if self.name == 'test':
            with open('./bad_case_input.txt', 'a', encoding='utf-8') as f:
                f.write(query_class + '\t' + str(qurey_index))
                f.write('\n')
                f.write(str(target_classes))
                f.write('\n')

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]) - 1)), self.K, False)
            for j in indices:
                if i == self.N - 1 and qurey_index == j:
                    if j > 0:
                        j -= 1
                    else:
                        j += 1
                word, mask, sbj, obj = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                mask = torch.tensor(mask).long()
                self.__additem__(support_set, word, mask)
                if self.use_img:
                    sbj_face, obj_face = self.__getface__(sbj, obj)
                    self.__addface__(support_set, sbj_face, obj_face)

        return support_set, query_set, query_label

    def __old_getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)  # 每个batch采样N个class
        support_set = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}
        query_set = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            if self.name == 'test' or self.name == 'val':
                # 统计下 sbj_obj list, support 和 query 的样本不能对应同一个三元组
                sbj_obj_list = {}
                for entry_data in self.json_data[class_name]:
                    sbj = entry_data['sbj']
                    obj = entry_data['obj']
                    sbj_obj = sbj + '-' + obj
                    if sbj_obj not in sbj_obj_list:
                        sbj_obj_list[sbj_obj] = [entry_data]
                    else:
                        sbj_obj_list[sbj_obj].append(entry_data)

                nrof_sbj_obj = len(sbj_obj_list.keys())
                all_sbj_obj = list(sbj_obj_list.keys())
                if self.K + self.Q <= nrof_sbj_obj:
                    indices = np.random.choice(list(range(nrof_sbj_obj)), self.K + self.Q, False)
                else:
                    indices = []
                    query_pair = np.random.choice(list(range(nrof_sbj_obj)), self.Q, False)
                    remain_pair = [i for i in list(range(nrof_sbj_obj)) if i not in query_pair]
                    try:
                        support_pair = np.random.choice(remain_pair, self.K, True)
                    except:
                        assert False, all_sbj_obj[query_pair[0]]
                    indices.extend(support_pair)
                    indices.extend(query_pair)

                count = 0
                for j in indices:
                    nrof_samples = len(sbj_obj_list[all_sbj_obj[j]])
                    ramdom_sample = random.randint(0, nrof_samples - 1)
                    word, mask, sbj, obj = self.__getraw__(sbj_obj_list[all_sbj_obj[j]][ramdom_sample])
                    word = torch.tensor(word).long()
                    mask = torch.tensor(mask).long()

                    for _ in range(self.multi_choose):
                        if count < self.K:
                            self.__additem__(support_set, word, mask)
                        else:
                            self.__additem__(query_set, word, mask)

                        if self.use_img:
                            sbj_face, obj_face = self.__getface__(sbj, obj)
                            if count < self.K:
                                self.__addface__(support_set, sbj_face, obj_face)
                            else:
                                self.__addface__(query_set, sbj_face, obj_face)
                    count += 1

            else:
                indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
                count = 0
                for j in indices:
                    word, mask, sbj, obj = self.__getraw__(self.json_data[class_name][j])
                    word = torch.tensor(word).long()
                    mask = torch.tensor(mask).long()

                    if count < self.K:
                        self.__additem__(support_set, word, mask)
                    else:
                        self.__additem__(query_set, word, mask)

                    if self.use_img:
                        sbj_face, obj_face = self.__getface__(sbj, obj)
                        if count < self.K:
                            self.__addface__(support_set, sbj_face, obj_face)
                        else:
                            self.__addface__(query_set, sbj_face, obj_face)
                    count += 1

            if self.name == 'test' or self.name == 'val':
                query_label += [i] * self.Q * self.multi_choose
            else:
                query_label += [i] * self.Q

        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}
    batch_query = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:  # k 对应的是 key
            if len(support_sets[i][k]) == 0:
                continue
            batch_support[k] += support_sets[i][k]  # 同一个的batch都放在同一个list中
        for k in query_sets[i]:
            if len(query_sets[i][k]) == 0:
                continue
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    batch_support = {key: value for key, value in batch_support.items() if len(value) > 0}
    batch_query = {key: value for key, value in batch_query.items() if len(value) > 0}
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label

def get_loader(name, encoder, N, K, Q, batch_size, 
        num_workers=1, collate_fn=collate_fn, root='./data', use_img=False, differ_scene=False, multi_choose=1):
    '''
    name: 数据集的名字
    '''
    dataset = FewRelDataset(name, encoder, N, K, Q, root, use_img, differ_scene, multi_choose)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

def prewhiten(x):
    # x = x.cpu().numpy()
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    # y = torch.from_numpy(y)
    return y