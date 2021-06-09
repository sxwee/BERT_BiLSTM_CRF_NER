import torch
from .bert_bilstm_crf import Bert_BiLSTM_CRF
from transformers import BertTokenizer
import pandas as pd

# 标签类别
VOCAB = ('O','B-DRUG','I-DRUG','B-MEDICINE','I-MEDICINE','B-CONTENT','I-CONTENT',
        'B-REASON','I-REASON','B-FREQ','I-FREQ','B-TIME','I-TIME','B-SDOSE','I-SDOSE','B-ARL','I-ARL',
        'B-IL','I-IL','B-MRL','I-MRL','B-DISEASE','I-DISEASE','B-SYMPTOM','I-SYMPTOM','B-ROA','I-ROA',
        'B-CROWD','I-CROWD','B-PE','I-PE','B-DSPEC','I-DSPEC')

# 标签到idx的映射
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
# idx到标签的映射
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

MODEL_PATH = 'ner/best.pt'
BERT_PATH = 'ner/bert_model/finetuning_model'

def extractEntity(y,words):
    """
    功能：提取实体
    y：标签序列
    words：字符序列，与标签序列中的标签一一对应
    """
    entities,entity = [],None
    for idx, st in enumerate(y):
        if entity is None:
            if st.startswith('B'):
                entity = {}
                entity['start'] = idx
            else:
                continue
        else:
            if st == 'O':
                entity['end'] = idx
                entity['content'] = ''.join(words[entity['start'] : entity['end']])
                entity['label'] = y[entity['start']][2:]
                entities.append(entity)
                entity = None
            elif st.startswith('B'):
                entity['end'] = idx
                entity['content']  = ''.join(words[entity['start'] : entity['end']])
                entity['label'] = y[entity['start']][2:]
                entities.append(entity)
                entity = {}
                entity['start'] = idx
            else:
                continue
    return entities

class SeqencePrecictionModel(object):
    def __init__(self, model_path, pretrained_path, device='cpu',isBERT=True):
        self.isBERT = isBERT
        self.device = torch.device(device)
        # 创建并加载模型
        if self.isBERT:
            self.model = Bert_BiLSTM_CRF(target_size=len(tag2idx),pretrained_path=pretrained_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)

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
    pred_model = SeqencePrecictionModel(MODEL_PATH, BERT_PATH, 'cuda', True)
    entities = pred_model.predict(text)
    df = pd.DataFrame(entities)
    print(df)
    