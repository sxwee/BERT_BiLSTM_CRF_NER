import torch
from data_loader import tag2idx, idx2tag
from model.bert_bilstm_crf import Bert_BiLSTM_CRF
from model.embedding_bilstm_crf import Embedding_BiLSTM_CRF
from transformers import BertTokenizer
from data_loader import word2id
from metrics import extractEntity
import pandas as pd

MODEL_PATH = 'checkpoints/BertBiLSTMCRF_finetuning_2/best.pt'
BERT_PATH = 'bert_model/chinese_L-12_H-768_A-12'

data_path = "datas/processed/test/03001.txt"

class SeqencePrecictionModel(object):
    def __init__(self, model_path, pretrained_path, device='cpu',isBERT=True):
        self.isBERT = isBERT
        self.device = torch.device(device)
        # 创建并加载模型
        if self.isBERT:
            self.model = Bert_BiLSTM_CRF(target_size=len(tag2idx),pretrained_path=pretrained_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        else:
            self.model = Embedding_BiLSTM_CRF(target_size=len(tag2idx))

        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def build(self,content):
        """
        功能：转换医药说明书为标准的输入格式,并返回input_ids，words
        """
        x,words = [],[]
        maxlen = 0
        for line in content:
            # linux下的换行符为\r\n
            # line = line.replace('\n','\r\n')
            token = [char for char in line]
            words.extend(token)
            if self.isBERT:
                token = ['[CLS]'] + token + ['[SEP]']
                xx = self.tokenizer.encode(token,add_special_tokens=False)
            else:
                xx = [word2id.get(char,word2id["[UNK]"]) for char in line]
            x.append(xx)
            maxlen = max(len(xx),maxlen)
        
        return self.pad(x,maxlen),words
        
    def pad(self,batch,maxlen):
        """
        功能：对输入进行填充，使得同一个batch的长度相同
        batch：医药说明书各个句子组成的batch
        maxlen：batch中最长的句子的长度
        """
        x = [sample + [0] * (maxlen - len(sample)) for sample in batch]
        
        return torch.LongTensor(x)


    def predict(self,content):
        """
        功能：预测实体所属的标签
        content：list，预测的内容
        """
        input_ids,words = self.build(content)
        input_ids = input_ids.to(self.device)
        # 获取掩码
        mask = (input_ids != 0).to(self.device)
        y_hats = self.model.predict(input_ids,mask)
        preds = []
        for y_hat in y_hats:
            if self.isBERT:
                preds.extend([idx2tag.get(idx,'O') for idx in y_hat[1:-1]])
            else:
                preds.extend([idx2tag.get(idx,'O') for idx in y_hat])
        
        return extractEntity(preds,words)


if __name__ == "__main__":
    text = ["与用药和人群相关的关系###通用名称：小儿金丹片。###用法用量：口服，周岁一次2 片，周岁以下酌减，一日3 次。"]
    data_path = "datas/processed/test/03001.txt"
    with open(data_path,'r',encoding='utf-8') as fp:
        content = fp.readlines()
    pred_model = SeqencePrecictionModel(MODEL_PATH, BERT_PATH, 'cuda', True)
    entities = pred_model.predict(content)
    df = pd.DataFrame(entities)
    