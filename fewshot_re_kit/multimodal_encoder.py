import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TransformerBlock(nn.Module):

    def __init__(self, Q_size, K_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.hidden_size = 512
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=self.hidden_size)

        self.relu = nn.ReLU()
        self.Q_linear = nn.Linear(Q_size, self.hidden_size)
        self.K_linear = nn.Linear(K_size, self.hidden_size)
        self.V_linear = nn.Linear(K_size, self.hidden_size)

        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.Q_linear.weight)
        init.xavier_normal_(self.K_linear.weight)
        init.xavier_normal_(self.V_linear.weight)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, episilon=1e-8):
        '''
        :param Q: (batch_size, img_embedding_dim)
        :param K: (batch_size, cls_embedding_dim)
        :param V: (batch_size, cls_embedding_dim)
        :return: output: (batch_size, hidden_embedding_dim)
        '''
        Q = self.Q_linear(Q)
        K = self.K_linear(K)
        V = self.V_linear(V)

        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()

        Q_K = Q.mm(K.permute(1, 0)) / (torch.sqrt(dk) + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size)
        V_att = Q_K_score.mm(V)

        # Q + V_att -> 残差连接

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, hidden_embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            # X = V_att
            output = self.FFN(X) + X

        return output


class MultimodalEncoder(nn.Module):

    def __init__(self, sentence_encoder, face_encoder, cross_modality='concate'):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.face_encoder = face_encoder
        self.layer_norm = nn.LayerNorm(normalized_shape=768+1024)
        self.transformer = TransformerBlock(768, 1024, False)
        self.cross_modality = cross_modality
        # print(self)

    def forward(self, inputs):
        sent_emb = self.sentence_encoder(inputs)
        face_emb = self.face_encoder(inputs['sbj_face'], inputs['obj_face'])
        if self.cross_modality == 'concate':
            mmodal_emb = torch.cat((sent_emb, face_emb), 1)
            mmodal_emb = self.layer_norm(mmodal_emb)
        elif self.cross_modality == 'transformer':
            mmodal_emb = self.transformer(sent_emb, face_emb, face_emb)
        else:
            assert False, "Not Implement"
        return mmodal_emb