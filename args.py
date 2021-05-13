import argparse

parser = argparse.ArgumentParser()

# 数据集批大小
parser.add_argument("--batch_size", type=int, default=64)
# 学习率
parser.add_argument("--lr", type=float, default=0.0001)
# adam_epsilon
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
# 数据集迭代次数
parser.add_argument("--n_epochs", type=int, default=30)
# 模型保存路径
parser.add_argument("--logdir", type=str, default="checkpoints/BertBiLSTMCRF_dropout_gradclip_0.0001/")
# 训练集路径
parser.add_argument("--trainset", type=str, default="datas/processed/train.txt")
# 验证集路径
parser.add_argument("--validset", type=str, default="datas/processed/val.txt")
# 测试集路径
parser.add_argument("--testset", type=str, default="datas/processed/test.txt")
# 预训练Bert路径
parser.add_argument("--pretrained_path",type=str,default="bert_model/chinese_L-12_H-768_A-12")
# 模型
parser.add_argument("--model",type=str,default="BertIDCNNCRF",choices=["BertBiLSTMCRF","EmbeddingBiLSTMCRF","BertIDCNNCRF"])
# 是否使用BERT
parser.add_argument("--isBERT",type=bool,default=True)
# 是否使用外部的词嵌入向量
parser.add_argument("--extra_embedding",type=bool,default=True)
# 外部词嵌入向量文件路径，只有选择使用外部词嵌入设置该参数才有效
parser.add_argument("--embedding_path",type=str,default="datas/glove1.txt")
# dropout率
parser.add_argument("--dropout_prob", type=float, default=0.5)


hp = parser.parse_args()

if __name__ == "__main__":
    pass