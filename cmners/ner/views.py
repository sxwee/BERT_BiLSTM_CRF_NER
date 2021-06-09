import json
from django.http.response import StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .funcs import groupByLabel,entity_nums,getLabelDict,tagHighlight,file_iterator
from .predict import SeqencePrecictionModel
import pandas as pd
import os

pred_model = SeqencePrecictionModel(
    model_path='ner/best.pt', 
    pretrained_path='ner/bert_model/chinese_L-12_H-768_A-12', 
    device='cuda', 
    isBERT=True)

entity_set = pd.DataFrame

# Create your views here.
def show_nersystem(request):
    global entity_nums
    return render(request,'my_ner.html',{'entity_nums':list(entity_nums.values())})

@csrf_exempt
def recognize_flag(request):
    if request.method == 'POST':
        fileString = request.POST.get('fileString')
        contents=[]
        for c in fileString.split('\n'):
            contents.append(c + '\n')
        global pred_model
        entities = pred_model.predict(contents)
        label_dict = getLabelDict(entities)
        innerhtml = tagHighlight(fileString,label_dict)
        global entity_set
        entity_set = pd.DataFrame(entities)
        nums,labels = groupByLabel(entity_set)
        return HttpResponse(json.dumps({
            'entity_nums':json.dumps(nums),
            'entity_labels':json.dumps(labels),
            'fileString':innerhtml
            }))


def save_entity_set(request):
    if request.method == 'GET':
        save_dir = "ner/export_dirs/"
        fname = 'temp.txt'
        save_path = os.path.join(save_dir, fname)
        global entity_set
        entity_set.to_csv(save_path,index=False,encoding="utf-8")
        response = StreamingHttpResponse(file_iterator(save_path))
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename="{}"'.format(fname)
        return response
