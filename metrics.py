import pandas as pd
import numpy as np

# 实体标签
LABEL = ['CONTENT','REASON','TIME','DSPEC','MRL','SYMPTOM','ROA','PE','DRUG','FREQ',
        'MEDICINE','ARL','IL','DISEASE','CROWD','SDOSE']

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

def getEvaluation(entities_pred,entities_true,measure='strict'):
    """
    功能：根据预测序列和真实标签序列计算宏精确率、宏召回率，宏F1 Score
    entities_pred：list，预测实体集
    entities_true：list，真实标签集
    measure：strict——严格指标，relaxed——松弛指标
    """
    # 将实体集按标签分类
    if entities_pred:
        df_preds = pd.DataFrame(entities_pred).groupby('label')
    else:
        df_preds = pd.DataFrame
    # 将实体集按标签分类
    df_trues = pd.DataFrame(entities_true).groupby('label')
    p,r = 0,0
    # 按类别获取percision,recall, f1_score
    for label in LABEL:
        try:
            group_pred = df_preds.get_group(label).values.tolist()
        except Exception:
            group_pred = []
        try:
            group_true = df_trues.get_group(label).values.tolist()
        except Exception:
            group_true = []
        
         # 严格指标下被识别为正例的真实正例
        if measure == 'strict':
            num_correct = len(set((tuple(i) for i in group_pred)) & set((tuple(i) for i in group_true)))
        elif measure == 'relaxed':
            # 松弛指标下被识别为正例的真实正例
            num_correct = 0
            n = len(group_true)
            for entity1 in group_pred:
                left = 0
                right = n - 1
                while left <= right:
                    mid = left + (right - left) // 2
                    entity2 = group_true[mid]
                    if max(entity1[0],entity2[0]) <= min(entity1[1] - 1,entity2[1] - 1):
                        # if entity1[2] != entity2[2]:print(entity1[2],entity2[2])
                        num_correct += 1
                        break
                    elif entity1[1] <= entity2[0]:
                        right = mid - 1
                    else:
                        left = mid + 1 
                # for entity2 in group_true:
                #     if max(entity1[0],entity2[0]) <= min(entity1[1] - 1,entity2[1] - 1):
                #         num_correct += 1
                #         break

        # 被识别为正例的样本总数
        num_proposed = len(group_pred)
        # 真实正例的总数
        num_gold = len(group_true)

        try:
            precision = num_correct / num_proposed
        except ZeroDivisionError:
            precision = 1.0

        try:
            recall = num_correct / num_gold
        except ZeroDivisionError:
            recall = 1.0

        try:
            f1 = 2*precision*recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 1.0 if precision*recall == 0 else 0

        p += precision
        r += recall
        print('label：{}\tprecision：{:.3f}\trecall：{:.3f}\tf1：{:.3f}'.format(label,precision,recall,f1))
        
    # 计算宏f1 score
    n = len(LABEL)
    macro_p = p / n
    macro_r = r / n
    try:
        macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)
    except ZeroDivisionError:
        macro_f1 = 1.0 if macro_p * macro_r == 0 else 0
        
    print("macro_P：{:.3f}\tmacro_R：{:.3f}\tmacro_F1：{:.3f}".format(macro_p, macro_r, macro_f1))

    return macro_p, macro_r, macro_f1

if __name__ == "__main__":
    words = np.array([line.split('\t')[0] for line in open(r"checkpoints\ner_predict.utf8", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_true =  np.array([line.split('\t')[1] for line in open(r"checkpoints\ner_predict.utf8", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred =  np.array([line.split('\t')[2] for line in open(r"checkpoints\ner_predict.utf8", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    # 加载预测集中的实体集
    entities_pred = extractEntity(y_pred,words)
    entities_true =  extractEntity(y_true,words)
    getEvaluation(entities_pred,entities_true,measure='strict')