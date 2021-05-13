import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from seqeval.metrics import f1_score
from transformers import BertModel,BertPreTrainedModel,BertConfig
from data_loader import DrugInstructionDataset, pad,tag2idx,idx2tag
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class BertClass(BertPreTrainedModel):
    def __init__(self,bert_model_path='chinese_L-12_H-768_A-12',dropout_prob=0.5):
        self.config = BertConfig.from_json_file(os.path.join(bert_model_path,'config.json'))
        super(BertClass,self).__init__(self.config)
        self.num_labels = len(tag2idx)
        self.hidden_dim = 768
        self.bert = BertModel.from_pretrained(bert_model_path)
        # self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
    
    def forward(self, ids, mask, labels):
        outputs = self.bert(ids)
        sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        pred = torch.argmax(emissions, dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
        
        return loss,pred

def train(model, train_iterator, valid_iterator, optimizer, fepoch=3):
    """
    训练函数
    """
    model.train()
    best_f1 = 0
    for epoch in range(fepoch):
        print("=============epoch {}=============".format(epoch + 1))
        total_loss = 0
        for i, batch in enumerate(train_iterator):
            ids,targets,_,_,_ = batch
            mask = ids!=0
            ids,targets,mask = ids.cuda(non_blocking=True),targets.cuda(non_blocking=True),mask.cuda(non_blocking=True)
            loss,_ = model(ids, mask, targets)
            # loss = output[0]
            total_loss += loss.item()
            if i % 100 == 0:
                print("step {} loss {}".format(i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
        
        print("Average training loss: ", total_loss / len(train_iterator))
        print("=============evaluate=============")
        cur_f1 = evaluate(model,valid_iterator)
        print('cur_f1: {} best_f1: {}'.format(cur_f1,best_f1))
        if cur_f1 - best_f1 > 0.0001:
            best_f1 = cur_f1
            # model_2_save = model.module.bert.module if hasattr(model.module.bert, "module") else model.module
            model.save_pretrained(save_directory='finetuning_model/')

def evaluate(model, iterator):
    model.eval()
    predictions , true_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            ids,targets,_,_,_ = batch
            mask = ids!=0
            ids,targets,mask = ids.cuda(non_blocking=True),targets.cuda(non_blocking=True),mask.cuda(non_blocking=True)
            _, pred = model(ids, mask, targets)
            predictions.extend(pred.cpu().numpy().flatten().tolist())
            true_labels.extend(targets.cpu().numpy().flatten().tolist())
        predictions = [idx2tag[idx] for idx in predictions]
        true_labels = [idx2tag[idx] for idx in true_labels]
        f1 = f1_score([true_labels],[predictions]) 
    return f1
           
if __name__ == "__main__":
    # 定义学习率
    lr = 2e-5
    # Bert 预训练模型
    pretrain_bert_path = "chinese_L-12_H-768_A-12"
    # 迭代的epoch数
    epoch = 4
    # 使用的GPU设备id
    device_ids = [0,1]
    # 定义模型
    model = BertClass()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids = device_ids)
        model = model.cuda()
    else:
        model = model.cuda()

    # 定义优化器
    # optimizer = AdamW(
    #     optimizer_grouped_parameters,
    #     lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #     eps=1e-8,  # args.adam_epsilon - default is 1e-8.
    # )   
    optimizer = optim.AdamW(params=model.parameters(),lr=lr,eps=1e-8)
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    print('Initial model Done')
    train_dataset = DrugInstructionDataset(data_path='datas/train.txt',
                    pretrained_path=pretrain_bert_path,
                    isBERT=True)
    
    val_dataset = DrugInstructionDataset(data_path='datas/val.txt',
                pretrained_path=pretrain_bert_path,
                isBERT=True)
    
    print('Load Data Done')
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)

    val_iter = data.DataLoader(dataset=val_dataset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    # 定义调度器
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps= epoch * len(train_iter)
    # )

    print('Start Train')
    train(model, train_iter, val_iter ,optimizer, epoch)


