import os
import torch
import torch.nn as nn
from args import hp
import torch.optim as optim
from torch.utils import data 
import numpy as np
from tqdm import tqdm
from model.bert_bilstm_crf import Bert_BiLSTM_CRF
from model.embedding_bilstm_crf import Embedding_BiLSTM_CRF
from model.bert_idcnn_crf import Bert_IDCNN_CRF
from analysis import drawPlot
from metrics import extractEntity,getEvaluation
from data_loader import DrugInstructionDataset, pad,tag2idx,idx2tag
import datetime

def train(model, iterator, optimizer, device, scheduler):
    """
    功能：模型训练
    model：待训练的模型
    iterator：划分为batch后的数据集迭代器
    optimizer：优化器
    device：计算设备，cpu或gpu设备
    scheduler：学习率调度器
    """
    model.train()
    # 计算整个epoch的平均损失
    sum_l,count = 0,0
    for i, batch in enumerate(iterator):
        x,y,_,_,_ = batch
        mask = x!=0
        # 数据放置到相应的设备上
        x,y,mask = x.to(device),y.to(device),mask.to(device)
        loss = model(x, y, mask)
        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()
        if scheduler:
            scheduler.step()
        sum_l += loss.item()
        count += 1

        if i % 50 == 0:
            print(f"step: {i}, loss: {loss.item()}")
        
    return sum_l / count

def eval(model, iterator, f, device,measure="strict"):
    """
    功能：模型的评估
    model：训练过的模型
    iterator：划分为batch后的数据集迭代器
    f：中间文件名
    device：数据集的处理设备
    measures：relaxed——松弛指标, strict——严格指标, both——二者
    """
    model.eval()
    Words, Tags, Y, Y_hat = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(iterator):
            x,y,words,tags,_ = batch
            x = x.to(device)
            mask = (x != 0).to(device)
            y_hat = model.predict(x,mask)
            Words.extend(words)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat)

    ## 暂存结果
    with open("temp", 'w', encoding='utf-8') as fout:
        for words,tags, y_hat in zip(Words, Tags, Y_hat):
            preds = [idx2tag.get(hat,'O') for hat in y_hat]
            if "Bert" in hp.model:
                for w, t, p in zip(words[1:-1], tags[1:-1], preds[1:-1]):
                    fout.write(f"{w}\t{t}\t{p}\n")
            else:
                for w, t, p in zip(words, tags, preds):
                    fout.write(f"{w}\t{t}\t{p}\n")
            fout.write("\n")

    words = np.array([line.split('\t')[0] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_true =  np.array([line.split('\t')[1] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred =  np.array([line.split('\t')[2] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    # 加载预测集中的实体集
    entities_pred = extractEntity(y_pred,words)
    # 加载训练集中的实体集
    entities_ture = extractEntity(y_true,words)
    # 计算评估指标（严格）
    macro_p, macro_r, macro_f1 = getEvaluation(entities_pred,entities_ture,measure=measure)

    final = f + ".P%.3f_R%.3f_F%.3f" %(macro_p, macro_r, macro_f1)

    with open(final, 'w', encoding='utf-8') as fout:
        result = open("temp", "r", encoding='utf-8').read()
        fout.write(f"{result}\n")

    os.remove("temp")

    return macro_p, macro_r, macro_f1
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    if hp.model == "BertBiLSTMCRF":  
        print('model: {} bert model：{}'.format(hp.model, hp.pretrained_path))
        model = Bert_BiLSTM_CRF(
            target_size=len(tag2idx),
            pretrained_path=hp.pretrained_path,
            dropout_prob=hp.dropout_prob
            ).to(device)
    elif hp.model == "BertIDCNNCRF":
        print('model: {} bert model：{}'.format(hp.model, hp.pretrained_path))
        model = Bert_IDCNN_CRF(
            target_size=len(tag2idx),
            pretrained_path=hp.pretrained_path,
            dropout_prob=hp.dropout_prob
            ).to(device)
    elif hp.model == "EmbeddingBiLSTMCRF":
        print("model：{} embedding_path: {} extra_embedding: {}".format(hp.model, hp.embedding_path,hp.extra_embedding))
        model = Embedding_BiLSTM_CRF(
            target_size=len(tag2idx),
            embedding_path=hp.embedding_path,
            dropout_prob=hp.dropout_prob, 
            extra_ebmedding=hp.extra_embedding
            ).to(device)
        hp.isBERT = False

    print('Initial model Done')
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
    # test_dataset = DrugInstructionDataset(data_path=hp.testset,pretrained_path=hp.pretrained_path)
    
    print('Load Data Done')
    train_iter = data.DataLoader(
        dataset=train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=pad,
        pin_memory=True
    )
    
    eval_iter = data.DataLoader(
        dataset=eval_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=pad,
        pin_memory=True 
    )

    optimizer = optim.AdamW(model.parameters(), lr = hp.lr,eps=hp.adam_epsilon)
    scheduler = None

    print('Start Train')
    precisions,recalls,f1s = [],[],[]
    losses = []
    best_result = 0
    for epoch in range(1, hp.n_epochs + 1):
        starttime = datetime.datetime.now()
        loss = train(model, train_iter, optimizer, device, scheduler)
        endtime = datetime.datetime.now()
        print('training time：{}s'.format((endtime - starttime).seconds))

        print("=========eval at epoch={}=========".format(epoch))
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname, device)
        losses.append(loss)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print("best f1: %.3f, current f1: %.3f" % (best_result, f1))
        # 保存训练完的模型
        if f1 - best_result > 2e-5:
            best_result = f1
            torch.save(model.state_dict(),hp.logdir + 'best.pt')
            print("weights were saved to {}".format(hp.logdir + 'best.pt'))

    # 绘制每个epoch的平均损失
    drawPlot([losses],"Batch{}_LR{}_Loss".format(hp.batch_size,hp.lr),
            "learning rate = {} batch size = {}".format(hp.lr,hp.batch_size))