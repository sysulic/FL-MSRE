import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from scipy import misc

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, root, use_img=False):
        self.root = root
        self.name = name
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert False

        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
        self.use_img = use_img
        if self.use_img:
            img_path = os.path.join(root, "entity_pair_in_img.json")
            img_info_path = os.path.join(root, "img_info.json")
            self.img_dir = os.path.join(root, 'imgs')
            self.img_data = json.load(open(img_path))
            self.img_info = json.load(open(img_info_path))

    def __getraw__(self, item):
        word, mask = self.encoder.tokenize(item['mask_sentence'])
        return word, mask, item['sbj'], item['obj']

    def __get_average_face__(self, img_list, name):
        nrof_imgs = len(img_list)
        average_face = torch.zeros(nrof_imgs, 3, 160, 160)
        for i, img in enumerate(img_list):
            bbox = self.img_info[img][name]
            bbox = [int(x) for x in bbox]
            full_img = misc.imread(os.path.expanduser(os.path.join(self.img_dir, img)))
            cropped = full_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            aligned = misc.imresize(cropped, (160, 160), interp='bilinear')  # 将人脸图 resize 到 160*160
            prewhitened = prewhiten(aligned)
            prewhitened = np.transpose(prewhitened, (2, 0, 1))
            face = torch.from_numpy(np.float32(prewhitened))
            average_face[i] = face

        average_face = torch.mean(average_face, dim=0)
        return average_face


    def __getface__(self, sbj, obj, differ_img=False):
        if differ_img:
            # 不同场景图
            sbj_img_list = [img for img_list in self.img_data[sbj].values() for img in img_list]
            random_sbj_img = random.sample(sbj_img_list, 1)[0]
            try:
                obj_img_list = [img for img_list in self.img_data[obj].values() for img in img_list if img != random_sbj_img]
            except:
                obj_img_list = [img for s in self.img_data.keys() for o, img_list in self.img_data[s].items() for img in img_list if
                                img != random_sbj_img and o == obj]
            if len(obj_img_list) == 0:
                obj_img_list = [random_sbj_img]

            random_obj_img = random.sample(obj_img_list, 1)[0]

        else:
            # random_img = random.sample(self.img_data[sbj][obj], 1)[0]  # 随机抽取一张图片
            # random_sbj_img = random_img
            # random_obj_img = random_img

            # 取所有人脸的均值
            sbj_face = self.__get_average_face__(self.img_data[sbj][obj], sbj)
            obj_face = self.__get_average_face__(self.img_data[sbj][obj], obj)
            assert False, (sbj_face.shape, obj_face.shape)


        sbj_bbox = self.img_info[random_sbj_img][sbj]
        sbj_bbox = [int(x) for x in sbj_bbox]
        obj_bbox = self.img_info[random_obj_img][obj]
        obj_bbox = [int(x) for x in obj_bbox]

        sbj_img = misc.imread(os.path.expanduser(os.path.join(self.img_dir, random_sbj_img)))
        sbj_cropped = sbj_img[sbj_bbox[1]:sbj_bbox[3], sbj_bbox[0]:sbj_bbox[2], :]
        aligned = misc.imresize(sbj_cropped, (160, 160), interp='bilinear')  # 将人脸图 resize 到 160*160
        prewhitened = prewhiten(aligned)
        prewhitened = np.transpose(prewhitened, (2, 0, 1))
        sbj_face = torch.from_numpy(np.float32(prewhitened))

        obj_img = misc.imread(os.path.expanduser(os.path.join(self.img_dir, random_obj_img)))
        obj_cropped = obj_img[obj_bbox[1]:obj_bbox[3], obj_bbox[0]:obj_bbox[2], :]
        aligned = misc.imresize(obj_cropped, (160, 160), interp='bilinear')
        prewhitened = prewhiten(aligned)
        prewhitened = np.transpose(prewhitened, (2, 0, 1))
        obj_face = torch.from_numpy(np.float32(prewhitened))

        return sbj_face, obj_face


    def __additem__(self, d, word, mask):
        d['word'].append(word)
        d['mask'].append(mask)

    def __addface__(self, d, sbj_face, obj_face):
        d['sbj_face'].append(sbj_face)
        d['obj_face'].append(obj_face)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)  # 每个batch采样N个class
        support_set = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}
        query_set = {'word': [], 'mask': [], 'sbj_face': [], 'obj_face': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            if self.name == 'test':
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
                    # while len(indices) < self.K + self.Q:
                    #     choose = np.random.choice(list(range(nrof_sbj_obj)), nrof_sbj_obj, False)
                    #     indices.extend(choose)
                    # indices = indices[:self.K + self.Q]
                    # repeat = [indices.count(x) for x in indices]
                    # indices_info = [(indices[k], repeat[k]) for k in range(len(indices))]
                    # indices_info = sorted(indices_info, key=lambda x:x[1], reverse=True)
                    # indices = [k[0] for k in indices_info]

                    query_pair = np.random.choice(list(range(nrof_sbj_obj)), self.Q, False)
                    remain_pair = [i for i in list(range(nrof_sbj_obj)) if i not in query_pair]
                    support_pair = np.random.choice(remain_pair, self.K, True)
                    indices.extend(support_pair)
                    indices.extend(query_pair)

                count = 0
                for j in indices:
                    nrof_samples = len(sbj_obj_list[all_sbj_obj[j]])
                    ramdom_sample = random.randint(0, nrof_samples - 1)
                    word, mask, sbj, obj = self.__getraw__(sbj_obj_list[all_sbj_obj[j]][ramdom_sample])
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
            batch_support[k] += support_sets[i][k]  # 不同的batch放在同一个list中
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
        num_workers=8, collate_fn=collate_fn, root='./data', use_img=False):
    '''
    name: 数据集的名字
    '''
    dataset = FewRelDataset(name, encoder, N, K, Q, root, use_img)
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