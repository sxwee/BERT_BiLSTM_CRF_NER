# 初始化实体数量
entity_nums={
    'CONTENT':10,
    'REASON':10,
    'TIME':10,
    'DSPEC':10,
    'MRL':10,
    'SYMPTOM':10,
    'ROA':10,
    'PE':10,
    'DRUG':10,
    'FREQ':10,
    'MEDICINE':10,
    'SDOSE':10,
    'ARL':10,
    'IL':10,
    'DISEASE':10,
    'CROWD':10
}
entity_labels = ['CONTENT','REASON','TIME','DSPEC','MRL','SYMPTOM','ROA','PE',
                    'DRUG','FREQ','MEDICINE','SDOSE','ARL','IL','DISEASE','CROWD']
# 实体颜色对应字典
entity_colors = {
            'CONTENT':'#6495ED',
            'REASON':'#8470FF',
            'TIME':'#00FFFF',
            'DSPEC':'#7FFFD4',
            'MRL':'#00FA9A',
            'SYMPTOM':'#FA8072',
            'ROA':'#FF69B4',
            'PE':'#A020F0',
            'DRUG':'#008B8B',
            'FREQ':'#FF4500',
            'MEDICINE':'#FF7F24',
            'SDOSE':'#FFFF00',
            'ARL':'#836FFF',
            'IL':'#00BFFF',
            'DISEASE':'#EE8262',
            'CROWD':'#FF34B3'
}

def groupByLabel(df):
    """
    功能：统计各个类别实体的数量
    df：dataframe，每行为一个实体
    """
    classes = df[['label','content']].groupby('label').count().sort_values(by='content',ascending=False)
    nums = classes['content'].tolist()  
    labels = classes.index.tolist()

    return nums,labels

def getLabelDict(entities):
    """
    功能：根据识别实体的位置获取位置标签字典
    entities：识别的实体集
    """
    label_dict = {}
    for entity in entities:
        start,end = entity['start'],entity['end']
        for i in range(start,end):
            if i == start:
                label_dict[i] = 'B-' + entity['label']
            else:
                label_dict[i] = 'I-' + entity['label']
    
    return label_dict

def tagHighlight(fileString,label_dict):
    """
    功能：根据实体集中实体的位置对文本内容进行高亮显示
    fileString：识别的内容
    labed_dict：位置标签字典
    """
    innerhtml = ""
    for i,ch in enumerate(fileString):
        chlabel = label_dict.get(i)
        if chlabel:
            if chlabel.startswith('B-'):
                innerhtml += ("<mark style='background-color:{}'>".format(entity_colors[chlabel[2:]]) + ch)
            else:
                nextlabel = label_dict.get(i + 1)
                if nextlabel:
                    innerhtml += ch
                    if nextlabel.startswith('B-'):
                        innerhtml += '</mark>'
                else:
                    innerhtml += (ch + '</mark>')
        else:
            innerhtml += ch
        if ch == '\n':innerhtml += '<br>'
    
    return innerhtml

def file_iterator(file_name, chunk_size=512):
    with open(file_name,encoding='utf-8') as f:
        while True:
            c = f.read(chunk_size)
            if c:
                yield c
            else:
                break

if __name__ == "__main__":
    pass
    