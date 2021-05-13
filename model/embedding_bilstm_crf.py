import torch
import torch.nn as nn
import json
from torchcrf import CRF
import numpy as np

word2id = json.load(open("datas/word2id.json", "r", encoding="utf8"))


class Embedding_BiLSTM_CRF(nn.Module):
    def __init__(self, target_size, embedding_dim=300, hidden_dim=200, dropout_prob=0.5,extra_ebmedding=True,
                embedding_path='datas/word2vec.bin'):
        super(Embedding_BiLSTM_CRF, self).__init__()
        # 获取词典字数
        self.vocab_size = len(word2id)
        # 是否加载外部词嵌入向量
        if extra_ebmedding:
            embedding_matrix = self.build_embdding_matrix(word_dict=word2id,
                                embedding_path=embedding_path)
            # 转为TensorFloat
            embedding_weight = torch.from_numpy(embedding_matrix).float()
            # 嵌入层加载预定义权重
            self.word_embeds = nn.Embedding.from_pretrained(embedding_weight)
        else:
            # 随机初始化嵌入层权重
            self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
            nn.init.uniform_(self.word_embeds.weight)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout_prob)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim // 2, 
                            batch_first=True,
                            num_layers=1, 
                            bidirectional=True
                        )
        # 全连接层
        self.fc = nn.Linear(hidden_dim, target_size)
        # 定义CRF层
        self.crf = CRF(target_size,batch_first=True)

    def forward(self, x, y, mask):
        """
        x:  (batch_size, max_seq_length)
        return: (batch_size, max_seq_length, target_size)
        """
        # 通过LSTM层获取特征
        emissions = self.get_lstm_features(x)
        loss = -self.crf(emissions=emissions, tags=y, mask=mask)
        return loss
    
    def get_lstm_features(self, x):
        """
        x: Bert模型中的input ids
        """
        embeds = self.word_embeds(x)
        # 过LSTM层
        enc, _ = self.lstm(embeds)
        # 过Dropout层
        enc = self.dropout(enc)
        # 过FC层
        lstm_feats = self.fc(enc)

        return lstm_feats
    
    
    def predict(self,x,mask=None):
        """
        预测函数
        x：Bert模型中的input ids
        """
        emissions = self.get_lstm_features(x)
        preds = self.crf.decode(emissions, mask)
        # preds = [seq + [-1]*(mask.size(1)-len(seq)) for seq in preds]
        return preds
    

    def load_pretrained_embedding(self,embedding_path):
        """
        功能：加载预训练词向量
        file_path：词嵌入向量路径
        """
        embedding_dict = {}
        with open(embedding_path,'r',encoding='utf-8') as fp:
            for line in fp.readlines():
                values = line.strip().split(' ')
                if len(values) < 300:continue
                word = values[0]
                embedding_vec = np.asarray(values[1:],dtype=np.float32)
                embedding_dict[word] = embedding_vec
        # print('Found %s word vectors.' % len(embedding_dict))
        return embedding_dict

    def build_embdding_matrix(self,word_dict,embedding_path,embedding_dim=300):
        """
        加载词向量矩阵
        word_dict：根据数据集构造的单词词典
        embedding_dim：词嵌入向量的维度
        """
        embedding_dict= self.load_pretrained_embedding(embedding_path)
        vocab_size = len(word_dict)
        count = 0
        embedding_matrix = np.zeros((vocab_size,embedding_dim))
        for word,i in word_dict.items():
            embedding_vec = embedding_dict.get(word)
            if embedding_vec is not None:
                embedding_matrix[i] = embedding_vec
            else:
                count += 1
        # print("loss word count: {}".format(count))
        return embedding_matrix
