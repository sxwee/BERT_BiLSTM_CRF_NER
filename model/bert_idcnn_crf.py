import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel,BertConfig
from model.idcnn import IDCNN

class Bert_IDCNN_CRF(nn.Module):
    def __init__(self, target_size, pretrained_path='bert_model/chinese_L-12_H-768_A-12',dropout_prob=0.5,filters=64):
        super(Bert_IDCNN_CRF, self).__init__()
        # 获取标签数
        self.tagset_size = target_size
        # 获取Bert预训练模型的配置文件
        self.config = BertConfig.from_pretrained(pretrained_path)
        # 获取隐藏层维数
        self.hidden_dim = self.config.hidden_size
        # 加载Bert预训练模型
        self.bert = BertModel.from_pretrained(pretrained_path)
        # 定义IDCNN层
        self.idcnn = IDCNN(input_size=self.hidden_dim, filters=filters)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout_prob)
        # 定义全连接层
        # self.linear = nn.Linear(64, 256)
        self.fc = nn.Linear(filters, self.tagset_size)
        # 定义CRF层
        self.crf = CRF(self.tagset_size,batch_first=True)

    def bert_enc(self, x):
        """
        通过Bert模型获取词嵌入向量
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, embedding_dim]
        """
        with torch.no_grad():
            enc = self.bert(x)[0]
        return enc

    def get_idcnn_features(self, x):
        """
        x: Bert模型中的input ids
        """
        embeds = self.bert_enc(x)
        # 过LSTM层
        idcnnout = self.idcnn(embeds)
        # 过Dropout层
        out = self.dropout(idcnnout)
        # 过FC层
        # out = self.linear(enc)
        out = self.fc(out)
        
        return out

    def forward(self, x, y, mask=None):
        """
        前向传播函数
        """
        # 获取发射矩阵得分
        emissions = self.get_idcnn_features(x)
        # 传入CRF层
        loss = -self.crf(emissions=emissions, tags=y, mask=mask)
        return loss

    def predict(self, x, mask=None):
        """
        预测函数
        x：Bert模型中的input ids
        """
        emissions = self.get_idcnn_features(x)
        preds = self.crf.decode(emissions, mask)
        # preds = [seq + [-1]*(mask.size(1)-len(seq)) for seq in preds]
        return preds
    
