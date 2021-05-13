import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from args import hp


# 标签类别
VOCAB = ('O','B-DRUG','I-DRUG','B-MEDICINE','I-MEDICINE','B-CONTENT','I-CONTENT',
        'B-REASON','I-REASON','B-FREQ','I-FREQ','B-TIME','I-TIME','B-SDOSE','I-SDOSE','B-ARL','I-ARL',
        'B-IL','I-IL','B-MRL','I-MRL','B-DISEASE','I-DISEASE','B-SYMPTOM','I-SYMPTOM','B-ROA','I-ROA',
        'B-CROWD','I-CROWD','B-PE','I-PE','B-DSPEC','I-DSPEC')

# 标签到idx的映射
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
# idx到标签的映射
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
# 加载单词词典
word2id = json.load(open("datas/word2id.json", "r", encoding="utf8"))

class DrugInstructionDataset(Dataset):
    def __init__(self,data_path,pretrained_path='bert_model/chinese_L-12_H-768_A-12',isBERT=True,maxlen=256) -> None:
        super().__init__()
        # 数据的路径
        self.data_path = data_path
        # 设置句子最大长度
        self.maxlen = maxlen
        # 获取句子和其对应的标签数组
        self.sents,self.tags_li = self.build()
        # 是否使用BERT的文本向量化方式
        self.isBERT = isBERT
        # 加载Bert预训练模型的Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        # 序列开始标签id
        self.cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        # 序列开始标签id
        self.sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")

    def build(self):
        """
        构造数据集
        """
        sents,tags_li = [],[]   
        words,tags = [],[]
        with open(self.data_path,'r',encoding='utf-8') as fp:
            for line in fp.readlines():
                line = line.rstrip().split('\t')
                if not line:continue
                char = line[0]
                cate = line[-1]
                # if char != '$':
                #     words.append(char)
                #     tags.append(cate)
                # if char=='$' and words:
                #     sents.append(words)
                #     tags_li.append(tags)
                #     words,tags = [],[]
                words.append(char)
                tags.append(cate)
                if char in ['。','?','!','！','？','；',';']:
                    sents.append(words)
                    tags_li.append(tags)
                    words,tags = [],[]
        return sents,tags_li
    
    def __getitem__(self, idx):
        """
        x：input_ids
        y：input_labels
        """
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        for w, t in zip(words, tags):
            if self.isBERT:
                xx = self.tokenizer.encode(w,add_special_tokens=False)
            else:
                xx = [word2id.get(w,word2id["[UNK]"])]
            yy = tag2idx[t]
            x.extend(xx)
            y.append(yy)
        if self.isBERT:
            x = [self.cls_id] + x + [self.sep_id]
            y = [0] + y + [0]
            words = ['[CLS]'] + words + ['[SEP]']
            tags = ['O'] + tags + ['O']
        seqlen = len(y)
        return x,y,words,tags,seqlen
    
    def __len__(self):
        return len(self.sents)


def pad(batch):
    """
    填充样本使得长度与batch中最长的样本一致
    """
    f = lambda x: [sample[x] for sample in batch]
    words,tags,seqlens = f(2),f(3),f(-1)
    maxlen = max(seqlens)
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: [PAD]
    x = f(0, maxlen)
    y = f(1, maxlen)
    f = torch.LongTensor

    return f(x),f(y),words,tags,seqlens   


if __name__ == '__main__':
    train_dataset = DrugInstructionDataset(data_path='datas/processed/train.txt',isBERT=True)
    x,y,words,tags,seqlen = pad([ train_dataset[0], train_dataset[1]])
    print(words)
    print(tags)
    print(x)
    print(y)
    print(x != 0)
    train_dataset = DrugInstructionDataset(
        data_path=hp.trainset,
        pretrained_path=hp.pretrained_path,
        isBERT=hp.isBERT
        )
    eval_dataset = DrugInstructionDataset(
        data_path=hp.validset,
        pretrained_path=hp.pretrained_path,
        isBERT=hp.isBERT
    )
    test_dataset = DrugInstructionDataset(
        data_path=hp.testset,
        pretrained_path=hp.pretrained_path,
        isBERT=hp.isBERT
    )

    print(len(train_dataset),len(eval_dataset),len(test_dataset))
        