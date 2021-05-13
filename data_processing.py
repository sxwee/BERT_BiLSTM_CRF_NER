import os
import numpy as np
import shutil
from transfer_data import TransferData

def divsion(org_path,div_path,usevalid=True):
    """
    功能：划分数据集
    org_path：原始数据集
    div_path：划分后的存储路径
    usevalid：布尔值，决定是否划分出验证集。
        True：划分训练集、验证集与测试集，划分比例为6:2:2
        False：划分训练集和验证集，划分比例为8:2
    """
    datasets = []
    for fn in os.listdir(org_path):
        if not fn.endswith('.txt'):continue
        datasets.append(fn)
    if not os.path.exists(div_path):
        os.mkdir(div_path)
    # 打乱数据集
    length = len(datasets)
    indices = np.arange(0,length)
    np.random.shuffle(indices)


    
    if usevalid:
    # 按比例划分训练集、验证集与测试集
        train_idx,val_idx,test_idx = indices[:int(length * 0.6)], \
                                    indices[int(length * 0.6):int(length * 0.8)],indices[int(length * 0.8):]
        datasets = np.array(datasets)
        train_datasets,val_datasets,test_datasets = datasets[train_idx].tolist(),\
                                                datasets[val_idx].tolist(),datasets[test_idx].tolist()
        # print(len(train_datasets),len(val_datasets),len(test_datasets))
        # 将划分的数据集分开存储
        copyFile(train_datasets,org_path,os.path.join(div_path,'train/'))
        copyFile(val_datasets,org_path,os.path.join(div_path,'val/'))
        copyFile(test_datasets,org_path,os.path.join(div_path,'test/'))
    else:
        train_idx,test_idx = indices[:int(length * 0.8)], indices[int(length * 0.8):]
        datasets = np.array(datasets)
        train_datasets,test_datasets = datasets[train_idx].tolist(),datasets[test_idx].tolist()
        # print(len(train_datasets),len(test_datasets))
        # 将划分的数据集分开存储
        copyFile(train_datasets,org_path,os.path.join(div_path,'train/'))
        copyFile(test_datasets,org_path,os.path.join(div_path,'test/'))
    
def copyFile(fnames,src,dst):
    """
    功能：将文件从源路径拷贝到目标路径
    fnames：待拷贝的文件名
    src：源目录
    dst：目标目录
    """
    if not os.path.exists(dst):
        os.mkdir(dst)
    for fname in fnames:
        shutil.copyfile(os.path.join(src, fname), os.path.join(dst, fname))
        fname = fname.replace('.txt','.ann')
        shutil.copyfile(os.path.join(src, fname), os.path.join(dst, fname))

def write_file(sens, labels, file_name):
    """
    写入文件
    """
    assert len(sens)==len(labels)
    with open(file_name, "w", encoding="utf8") as f:
        for i in range(len(sens)):
            assert len(sens[i])==len(labels[i])
            for j in range(len(sens[i])):
                f.write(sens[i][j]+"\t"+labels[i][j]+"\n")
            f.write("\n")
    
    print(file_name + "'s datasize is " , len(sens))


def get_dict(data_path, filter_word_num):
    """
    功能：获取单词字典
    data_path：文件路径
    filter_word_num：词频最小值，出现次数低于该值的过滤掉
    """
    word_count = {}
    for dir in os.listdir(data_path):
        if dir.endswith('tar.gz'):continue
        for fname in os.listdir(os.path.join(data_path,dir)):
            if not fname.endswith('.txt'):continue
            with open(os.path.join(data_path,dir,fname),'r',encoding='utf-8') as fp:
                for word in fp.read():
                    word_count[word] = word_count.get(word, 0) + 1
        
    # 过滤低频词
    word2id = {
        "[PAD]": 0, 
        "[UNK]": 1
    }

    for word, count in word_count.items():
        if count >= filter_word_num:
            word2id[word] = len(word2id)
    
    print("Total %d tokens, filter count<%d tokens, save %d tokens."%(len(word_count)+2, filter_word_num, len(word2id)))

    return word2id

if __name__ == '__main__':
    org_path='datas/org_data/05/'
    div_path='datas/processed1/'
    usevalid=False
    divsion(org_path=org_path,div_path=div_path,usevalid=False)
    print('dataset divid done')
    if usevalid:
        handler = TransferData(data_path=os.path.join(div_path,'train'),save_path=os.path.join(div_path,'train.txt'))
        train_datas = handler.transfer()
        handler1 = TransferData(data_path=os.path.join(div_path,'val'),save_path=os.path.join(div_path,'val.txt'))
        train_datas = handler1.transfer()
        handler2 = TransferData(data_path=os.path.join(div_path,'test'),save_path=os.path.join(div_path,'test.txt'))
        train_datas = handler2.transfer()
    else:
        handler = TransferData(data_path=os.path.join(div_path,'train'),save_path=os.path.join(div_path,'train.txt'))
        train_datas = handler.transfer()
        handler2 = TransferData(data_path=os.path.join(div_path,'test'),save_path=os.path.join(div_path,'test.txt'))
        train_datas = handler2.transfer()
    print('data transorm BIO done')
