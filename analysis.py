import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import re
import numpy as np
from metrics import extractEntity,getEvaluation

def loadDataset(data_path):
    """
    功能：加载目录中的所有实体
    data_path：数据集文件的目录路径
    """
    label_dict = json.load(open('datas/classification.json','r',encoding='utf-8'))
    datas = []
    for fn in os.listdir(data_path):
        # 过滤原始文本数据
        if not fn.endswith('.ann'):continue
        with open(os.path.join(data_path,fn),'r',encoding='utf-8') as fp:
            for line in fp.readlines():
                # 过滤掉关系
                if line.startswith('R'):continue
                res = line.strip().split('	')
                label,startId,endId = res[1].split(' ')
                content = res[2]
                docId = fn.replace('.ann','')
                # 转换为一级分类标签
                label = label_dict.get(label)
                # 过滤未被考虑进内的部分实体
                if label:datas.append([docId,label,startId,endId,content])

    return datas

def groupByLabel(df):
    """
    功能：统计各个类别实体的数量
    df：dataframe，每行为一个实体
    """
    classes = df[['label','content']].groupby('label').count().sort_values(by='content',ascending=False)
    nums = classes['content'].tolist()  
    labels = classes.index.tolist()

    return nums,labels

def autolable(rects,gap=10):
    """
    功能：绘制柱形图的值
    gap：文字距离柱形的距离
    """
    for rect in rects:
        height = rect.get_height()
        if height>=0:
            plt.text(rect.get_x()+rect.get_width()/2.0 - 0.4,height + gap,'{}'.format(height))
        else:
            plt.text(rect.get_x()+rect.get_width()/2.0 - 0.4,height - gap,'{}'.format(height))
            # 如果存在小于0的数值，则画0刻度横向直线
            plt.axhline(y=0,color='black')

def drawBar(x,height,title='实体数量直方图',gap=10):
    """
    功能：绘制直方图
    x：柱形的横坐标
    height：柱形的高度
    title：图标题
    gap：直方图中文字距离柱形的距离
    """
    map_vir = cm.get_cmap(name='inferno')
    colors = map_vir(height)
    fig = plt.figure(figsize=(8,6.5))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.title(title)
    plt.xticks(rotation = 90)
    # plt.tick_params(axis='x', labelsize=8)  
    ax = plt.bar(x,height,color=colors,edgecolor='black')
    autolable(ax,gap)
    plt.savefig('images/{}.png'.format(title))

def drawPlot(heights,fname,title,labels=None,marker='o'):
    """
    功能：绘制折线图
    heights：纵坐标的值
    fname：折线图文件名
    title：折线图的标题
    labels：图例标签
    marker：设置折线图上的点形状
    """
    plt.figure()
    plt.title(title)
    x = [i for i in range(1,len(heights[0]) + 1)]
    plt.xlabel('epoch')
    # 设置横坐标的刻度间隔
    plt.xticks([i for i in range(0,len(heights[0]) + 1,5)])
    for h in heights:
        plt.plot(x,h,marker=marker)
    # 设置图例
    if labels:plt.legend(labels = labels)
    
    plt.savefig("images/{}.png".format(fname))


def drawPRF(batch,lr,dir_name,measures='both'):
    """
    功能：根据训练过程每个epoch保存的中间结果文件绘制PRF曲线
    batch,lr：训练的超级参数batch size和learning rate
    dir_name：中间文件的路径名
    measures：relaxed——松弛指标, strict——严格指标, both——二者
    """
    # 对文件按epoch排序
    fns = [fn for fn in os.listdir(os.path.join('checkpoints',dir_name)) if not fn.endswith('.pt') and not fn.startswith('test')]
    # 分别按照严格指标和松弛指标计算Percision,Recall,F1
    ps,rs,fs = [],[],[]
    ps1,rs1,fs1 = [],[],[]
    fns.sort(key=lambda x:int(re.search('(\d+).?',x).group(1)))
    for fn in fns:
        if fn.endswith('.pt'):continue
        # 获取中间文件的路径
        relative_path = os.path.join('checkpoints', dir_name, fn)
        # 计算查准率、查全率和F1
        s_metrics,r_metrics = calPRF(relative_path,measures)
        if s_metrics:
            ps.append(s_metrics[0])
            rs.append(s_metrics[1])
            fs.append(s_metrics[2])
        if r_metrics:
            ps1.append(r_metrics[0])
            rs1.append(r_metrics[1])
            fs1.append(r_metrics[2])

    if measures in ['both','strict']:
        drawPlot([ps,rs,fs],"{}_LR{}_P_R_F1_Strict".format(batch,lr,lr),
                title="Strict Metrics".format(lr),labels=["Percision","Recall","F1 Score"])
    
    if measures in ['both','relaxed']:
        drawPlot([ps1,rs1,fs1],"{}_LR{}_P_R_F1_Relaxed".format(batch,lr,lr),
            title="Relaxed Metrics".format(lr),labels=["Percision","Recall","F1 Score"])

def calPRF(file_path,measures):
    """
    功能：计算精确率，召回率和F1分数
    file_path：预测结果文件
    measures：relaxed——松弛指标, strict——严格指标, both——二者
    """
    s_metrics,r_metrics = None,None
    words = np.array([line.split('\t')[0] for line in open(file_path, 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_true =  np.array([line.split('\t')[1] for line in open(file_path, 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred =  np.array([line.split('\t')[2] for line in open(file_path, 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    entities_pred = extractEntity(y_pred,words)
    entities_ture = extractEntity(y_true,words)
    if measures in ['both','strict']:
        macro_p, macro_r, macro_f1 = getEvaluation(entities_pred,entities_ture,measure='strict')
        s_metrics = [macro_p, macro_r, macro_f1]
    if measures in ['both','relaxed']:
        macro_p, macro_r, macro_f1 = getEvaluation(entities_pred,entities_ture,measure='relaxed')
        r_metrics = [macro_p, macro_r, macro_f1]
    
    return s_metrics,r_metrics



if __name__ == "__main__":
    # datas = loadDataset(data_path='datas/org_data/07')
    # df = pd.DataFrame(datas,columns=['docId','label','startId','endId','content'])
    # print(df.shape)
    # nums,labels = groupByLabel(df)
    # drawBar(labels,nums,'实体数量直方图1')
    drawPRF(batch=64,lr=0.001,dir_name='BertBiLSTMCRF_finetuning_0.0001')
    