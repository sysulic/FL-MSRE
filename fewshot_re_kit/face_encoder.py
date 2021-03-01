import torch
from torch import nn
from . import network


class FacenetEncoder(nn.Module):
    def __init__(self):
        super(FacenetEncoder, self).__init__()
        self.embedding = network.inception_resnet_v1.InceptionResnetV1(pretrained='vggface2')

    def forward(self, sbj_face, obj_face):
        sbj_emb = self.embedding(sbj_face)
        obj_emb = self.embedding(obj_face)
        face_emb = torch.cat((sbj_emb, obj_emb), 1)  # [batch_size, 1024]
        return face_emb
