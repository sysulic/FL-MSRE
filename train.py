from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, RobertaSentenceEncoder
from fewshot_re_kit.multimodal_encoder import MultimodalEncoder
from fewshot_re_kit.face_encoder import FacenetEncoder
import models
from models.proto import Proto
from models.d import Discriminator
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_data',
            help='train file')
    parser.add_argument('--val', default='val_data',
            help='val file')
    parser.add_argument('--test', default='test_data',
            help='test file')
    parser.add_argument('--adv', default=None,
            help='adv file')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=1000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--repeat_test', default=8, type=int,
                        help='repeat test stage')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='bert',
            help='encoder: cnn or bert or roberta')
    parser.add_argument('--cross_modality', default='concate',
                        help='cross-modality module')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-1, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--multi_choose', default=1, type=int,
           help='choose multi random faces')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--root_data', default='./data',
                        help='the root path stores data')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
           help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
           help='concatenate entity representation as sentence rep')
    parser.add_argument('--use_img', action='store_true',
                        help='use img info')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')

    parser.add_argument('--differ_scene', action='store_true',
                        help='use face image in different scenes')


    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Multimodal Social Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-chinese'
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt or 'roberta-base'
        sentence_encoder = RobertaSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep)
    else:
        raise NotImplementedError

    train_data_loader = get_loader(opt.train, sentence_encoder,
            N=trainN, K=K, Q=Q, root=opt.root_data, batch_size=batch_size, use_img=opt.use_img, differ_scene=opt.differ_scene, multi_choose=opt.multi_choose)
    val_data_loader = get_loader(opt.val, sentence_encoder,
            N=N, K=K, Q=Q, root=opt.root_data, batch_size=batch_size, use_img=opt.use_img, differ_scene=opt.differ_scene, multi_choose=opt.multi_choose)
    test_data_loader = get_loader(opt.test, sentence_encoder,
            N=N, K=K, Q=Q, root=opt.root_data, batch_size=batch_size, use_img=opt.use_img, differ_scene=opt.differ_scene, multi_choose=opt.multi_choose)

   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)

    if opt.differ_scene:
        prefix = '-'.join(['differ', model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    else:
        prefix = '-'.join(['same', model_name, encoder_name, opt.train, opt.val, str(N), str(K)])

    if opt.use_img:
        prefix = 'img_' + prefix
    if opt.cross_modality == 'transformer':
        prefix = 'transformer' + '-' + prefix

    if opt.multi_choose > 1:
        prefix = 'choose-' + str(opt.multi_choose) + '-' + prefix

    prefix = '0227' + prefix
    
    if model_name == 'proto':
        if opt.use_img:
            face_encoder = FacenetEncoder()
            multimodal_encoder = MultimodalEncoder(sentence_encoder=sentence_encoder, face_encoder=face_encoder, cross_modality=opt.cross_modality)
            model = Proto(multimodal_encoder, dot=opt.dot)
        else:
            model = Proto(sentence_encoder, dot=opt.dot)
    else:
        raise NotImplementedError

    # if not os.path.exists('checkpoint'):
    #     os.mkdir('checkpoint')
    # ckpt = 'checkpoint/{}.pth.tar'.format(prefix)

    checkpoint_dir = os.path.join(opt.root_data, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    ckpt = os.path.join(checkpoint_dir, '{}.pth.tar'.format(prefix))
    

    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, grad_iter=opt.grad_iter, fp16=opt.fp16,
                pair=opt.pair, train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim,
                multi_choose=opt.multi_choose)
    else:
        ckpt = opt.load_ckpt

    result_txt = os.path.join(checkpoint_dir, '{}.txt'.format(opt.root_data.split('/')[1] + '_' + prefix))
    with open(result_txt, 'a', encoding='utf-8') as f:
        for _ in range (opt.repeat_test):
            acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair, multi_choose=opt.multi_choose)
            print("RESULT: %.2f" % (acc * 100))
            f.write(str(acc * 100))
            f.write('\n')


if __name__ == "__main__":
    main()
