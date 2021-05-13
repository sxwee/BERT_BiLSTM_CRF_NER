# BERT_BiLSTM_CRF_NER

**项目简介**：该项目用BERT预训练模型+双向长短期记忆网络+条件随机（BERT-BiLSTM-CRF）对中文医药说明书进行命名实体识别。

## 环境要求

**Python版本**：3.6.11

**深度学习框架**：Pytorch-1.7.0

**依赖库及其版本**

| 依赖库       | 版本   |
| ------------ | ------ |
| matplotlib   | 3.3.2  |
| pandas       | 1.1.4  |
| numpy        | 1.19.3 |
| transformers | 4.5.1  |
| tqdm         | 4.49.0 |

## 相关文件下载

BERT模型：[下载地址](https://pan.baidu.com/s/14Oe28mjHN29LnnNHTsjqmA)，提取码：kgap，备注：下载后解压放入bert_model目录下

Word2Vec和Glove词嵌入：[下载地址](https://pan.baidu.com/s/1hMZhXCxg1ApvgdtvyIKBAQ)，提取码：6xt0，备注：下载后解压放入datas目录下

原始数据集：[下载地址](https://pan.baidu.com/s/12SPmw0O-uWnLkOIpLsHABQ)，提取码：y0e1，备注：下载后解压放入datas目录下

# 使用方法

## 数据集预处理

数据集是使用Brat标注工具标注得来的，首先运行data_processing.py文件进行数据集的预处理，主要进行数据集的划分，BIO序列文件的获取。

## 模型训练

本项目实现了**EmbeddingBiLSTMCRF**、**BertBiLSTMCRF**、**BertIDCNNCRF**三个模型。其中EmbeddingBiLSTMCRF是指使用常规词嵌入+BiLSTM+CRF，其中常规词嵌入包括nn.Embedding层随机初始化的词嵌入或训练好的Word2Vec/Glove词嵌入。BertBiLSTMCRF是指使用Bert词嵌入+BiLSTM+CRF。而BertIDCNNCRF是指Bert词嵌入+IDCNN+CRF，其中IDCNN为自膨胀卷积模块。

默认开启BertBiLSTMCRF模型的训练，运行命令如下（后续的参数可根据自己的实际情况进行更改，也可不进行更改）：

```python
python main.py	--trainset  datas/processed/train.txt \
				--validset  datas/processed/val.txt \
    			--pretrained_path  bert_model/chinese_L-12_H-768_A-12 \
        		--model model_name BertBiLSTMCRF \
            	--isBERT True \
            	--extra_embedding False \
                --embedding_path datas/word2vec.bin \
                --dropout_prob 0.5 \
                --batch_size 64 \
                --lr 0.001
```

参数说明：

- **trainset**：训练集的路径
- **validset**：验证集的路径
- **pretrained_path**：bert模型的路径
- **model**：模型名，['EmbeddingBiLSTMCRF', 'BertBiLSTMCRF', 'BertIDCNNCRF']
- **isBERT**：布尔值，True表示使用**Bert模型**来加载序列数据，False表示采用**常规词嵌入的方式**来加载序列数据
- **extra_embedding**：布尔值，True表示使用外部词嵌入，False表示直接使用nn.Embedding随机初始化生成的词嵌入
- **embedding_path**：外部词嵌入的加载路径，前提是extra_embedding为True
- **dropout_prob**：设置dropout率，范围[0,1]
- **batch_size**：数据批大小
- **lr**：学习率

## 模型评估

打开test.py文件，设置model_path的值，即训练好的模型的路径，然后运行下面命令即可：

```python
python test.py 	--testset datas/processed/test.txt \
				--model BiLSTMCRF
```

参数说明：

testset：测试机的路径

**备注**：运行命名后面的参数也可先做args.py中设置好，然后直接运行相应的脚本即可。