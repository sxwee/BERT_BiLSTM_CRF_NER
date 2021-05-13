import torch
from main import eval,hp
import torch.utils.data as data
from data_loader import DrugInstructionDataset,pad,tag2idx
from model.bert_bilstm_crf import Bert_BiLSTM_CRF
from model.embedding_bilstm_crf import Embedding_BiLSTM_CRF
from model.bert_idcnn_crf import Bert_IDCNN_CRF

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    model_path = 'checkpoints/BertBiLSTMCRF_finetuning_0.0001/best.pt'
    if hp.model == "BertBiLSTMCRF":
        print('model: {} bert model：{}'.format(hp.model, hp.pretrained_path))
        model = Bert_BiLSTM_CRF(target_size=len(tag2idx),
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
        print("model：{} embedding_path: {} extra_embedding: {}".format(hp.model, hp.embedding_path, hp.extra_embedding))
        model = Embedding_BiLSTM_CRF(target_size=len(tag2idx), 
                                    embedding_path=hp.embedding_path,
                                    dropout_prob=hp.dropout_prob, 
                                    extra_ebmedding=hp.extra_embedding
                                    ).to(device)
        hp.isBERT = False
    # 加载预训练好的模型
    model.load_state_dict(torch.load(model_path))
    # 加载测试集
    test_dataset = DrugInstructionDataset(data_path=hp.testset, 
                                        pretrained_path=hp.pretrained_path, 
                                        isBERT=hp.isBERT
                                        )
    # 将测试集分为batchs
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad,
                                pin_memory=True)
    # 设置结果保存文件目录
    hp.logdir = model_path.replace('best.pt','')
    precision, recall, f1 = eval(model, test_iter, hp.logdir + 'test', device,"strict")
    precision, recall, f1 = eval(model, test_iter, hp.logdir + 'test', device,"relaxed")