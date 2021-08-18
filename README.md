# FL-MSRE

Code and data for our paper [FL-MSRE: A Few-Shot Learning based Approach to Multimodal Social Relation Extraction (AAAI 2021)]()

Our code is based on [FewRel](https://github.com/thunlp/FewRel) and [FaceNet-pytorch](https://github.com/timesler/facenet-pytorch)

## Prerequisites

- Pytorch 1.0 or higher, with NVIDIA CUDA Support
- Python 3.6
- pillow 7.1.2
- transformers 3.0.2
- scikit-learn  0.23.1
- scipy  1.2.1
- torchvision  0.4.2

## Dataset preparation

We constructed three datasets for multimodal social relation extraction. To replicate the experiments, you need to prepare your dataset as the following. Taking the `FC-TF` dataset as an example.

```
FC-TF
├── imgs
│   ├── hlm_xxxx_xxxx.jpg
│   ├── ……
│   └── xyj_xxxx_xxxx.jpg
├── entity_pair_in_img.json
├── img_info.json
├── train_data.json
├── val_data.json
└── test_data.json
```

You can download all images, and put them under the corresponding datasets for the [website](https://drive.google.com/file/d/1myHcRDA71LFwIIGhm5Fg_idgZHezA93O/view?usp=sharing).

## Pre-trained Checkpoint

Due to the large size, FaceNet and Bert pre-trained checkpoint are not included.

Please download the [FaceNet pre-trained checkpoint](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) and [Bert pre-trained checkpoint](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz) here, and put them in the `pretrained` under the root.

Here we provide [all checkpoints of FL-MSRE](https://drive.google.com/file/d/1kb3Vi50NdaAyQyExSTMkBUnPGx-9TnVX/view?usp=sharing).

## Code Structure
```
FL-MSRE
├── datasets
│   ├── FC-TF
│   ├── OM-TF
│   └── DRC-TF
├── fewshot_re_kit
├── models
├── pretrained
│   ├── vggface2
│   	├── 20180402-114759-vggface2-features.pt
│   	└──20180402-114759-vggface2-logits.pt
│   └──  bert_base_chinese
│   	├── config.json
│   	├── pytorch_model.bin
│   	└── vocab.txt
└── train.py
```


- train.py: Program Runner
- fewshot_re_kit
  - face_encoder.py: Face encoder, based on FaceNet
  - sentence_encoder.py: Sentence encoder, based on Bert
  - framework.py: Framework model of FL-MSRE model
- models
  - proto.py: Prototypcial Network

## Training

To train the baseline model with 3-way-1-shot:

```shell
python train.py --trainN 3 --N 3 --K 1 --Q 1 --model proto --encoder bert --hidden_size 768 --val_step 10000 --batch_size 2 --pretrain_ckpt {PRETRAINED_CKPT_PATH} --train_iter 20000 --grad_iter 4 --root_data {DATASET_PATH} --multi_choose 5
```

To train the FL-MSRE model with 3-way-1-shot, using the face images of entity pair from the same image:

```shell
python train.py --trainN 3 --N 3 --K 1 --Q 1 --model proto --encoder bert --hidden_size 768 --val_step 10000 --batch_size 2 --pretrain_ckpt {PRETRAINED_CKPT_PATH} --train_iter 20000 --grad_iter 4 --root_data {DATASET_PATH} --multi_choose 5 --use_img
```

The `--use_img` specifies that combing face images information for social relation extraction.

To train the FL-MSRE model with 3-way-1-shot, using the face images of the entity pair from different images:

```shell
python train.py --trainN 3 --N 3 --K 1 --Q 1 --model proto --encoder bert --hidden_size 768 --val_step 10000 --batch_size 2 --pretrain_ckpt {PRETRAINED_CKPT_PATH} --train_iter 20000 --grad_iter 4 --root_data {DATASET_PATH} --multi_choose 5 --use_img --differ_scene
```

The `--differ_scene` specifies that using the face images of  the entity pair from different images.

## Inference

After training, you can evaluate the baseline model with 3-way-1-shot:

```shell
python train.py --trainN 3 --N 3  --K 3 --Q 1 --model proto --encoder bert --hidden_size 768 --batch_size 2 --pretrain_ckpt {PRETRAINED_CKPT_PATH} --test_iter 10000 --load_ckpt {CKPT_PATH} --root_data {DATASET_PATH} --multi_choose 5 --only_test --use_img --differ_scene
```

The `--only_test` specifies that only evaluating the model.

You can also evaluate the FL-MSRE model with 3-way-1-shot:

```shell
python train.py --trainN 3 --N 3  --K 3 --Q 1 --model proto --encoder bert --hidden_size 768 --batch_size 2 --pretrain_ckpt {PRETRAINED_CKPT_PATH} --test_iter 10000 --load_ckpt {CKPT_PATH} --root_data {DATASET_PATH} --multi_choose 5 --use_img --only_test
```

```shell
python train.py --trainN 3 --N 3  --K 3 --Q 1 --model proto --encoder bert --hidden_size 768 --batch_size 2 --pretrain_ckpt {PRETRAINED_CKPT_PATH} --test_iter 10000 --load_ckpt {CKPT_PATH} --root_data {DATASET_PATH} --multi_choose 5 --use_img --differ_scene --only_test
```
