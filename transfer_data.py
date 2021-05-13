import os
import json

class TransferData:
    """
    将原始文档和标注文件转换为BIO序列文件
    """
    def __init__(self,data_path, save_path):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        
        self.label_dict = self.loadLabelDict()
        # 待处理原始药品说明书文件路径
        self.origin_path = os.path.join(cur, data_path)
        # 转换后的药品说明书数据存储路径
        self.bio_save_path = os.path.join(cur, save_path)

        
    def loadLabelDict(self):
        """
        功能：加载类别标签字典
        """
        with open('datas/classification.json','r',encoding='utf-8') as fp:
            label_dict = json.load(fp)
        return label_dict


    def transfer(self):
        """
        功能：格式转换，结果将得到一个BIO序列文件
        """
        with open(self.bio_save_path, 'w+', encoding='utf-8') as f:
            for root,_,files in os.walk(self.origin_path):
                for file in files:
                    # filepath为源文件
                    filepath = os.path.join(root, file)
                    if '.txt' not in filepath:continue
                    # label_path为标注文件路径
                    label_path = filepath.replace('.txt','.ann')
                    print("org path：{} label path：{}".format(filepath,label_path))
                    content = open(filepath, 'r', encoding='utf-8').read().strip()
                    res_dict = {}
                    for line in open(label_path, 'r', encoding='utf-8'):
                        # 过滤掉关系
                        if line.startswith('R'):continue
                        # 取出实体标签和在文本中的位置
                        res = line.strip().split('	')[1].split(' ')
                        label,start,end = res[0],int(res[1]),int(res[2])
                        label_id = self.label_dict.get(label)
                        # 过滤掉某些的实体类型
                        if label_id == None:continue
                        for i in range(start, end):
                            if i == start:
                                label_cate = 'B-' + label_id
                            else:
                                label_cate = 'I-' + label_id
                            res_dict[i] = label_cate
                    # linux下换行是\n但到windows下却是\r\n，这里替换为两个$$
                    content = content.replace('\n','  ')
                    # 遍历整个文本文件
                    for indx, char in enumerate(content):
                        char_label = res_dict.get(indx, 'O')
                        # 过滤空格和TAB
                        if char != ' ' and char != '　':
                            f.write(char + '\t' + char_label + '\n')
                            # if char in ['。','?','!','！','？']:f.write('\n')
               

if __name__ == '__main__':
    handler = TransferData(data_path='datas/processed/train',save_path='datas/processed/train.txt')
    train_datas = handler.transfer()
    handler1 = TransferData(data_path='datas/processed/test',save_path='datas/processed/test.txt')
    train_datas = handler1.transfer()
    handler2 = TransferData(data_path='datas/processed/val',save_path='datas/processed/val.txt')
    train_datas = handler2.transfer()
    print('data transorm BIO done')