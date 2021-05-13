import sys
import tkinter as tk
import pandas as pd
from tkinter import messagebox
import matplotlib.pyplot as plt
from tkinter.constants import INSERT, NO
from analysis import groupByLabel
from tkinter.filedialog import askopenfilename,asksaveasfilename
from predict import SeqencePrecictionModel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NERSystem():
    """
    中文命名实体识别系统GUI
    """
    def __init__(self,model_path='checkpoints/BertBiLSTMCRF_finetuning/best.pt',
                    bert_path='bert_model/chinese_L-12_H-768_A-12',isBERT=True) -> None:
        # 加载模型
        self.pred_model = SeqencePrecictionModel(model_path, bert_path, 'cuda', isBERT)
        # 初始化窗口
        self.window = tk.Tk()
        # 设置窗口名称
        self.window.title('中文命名实体识别系统')
        # 设置窗口大小
        self.window.geometry("1150x600")
        # 设置窗口无法改变
        self.window.resizable(False,False)
        # 设置窗口的背景颜色
        self.window.configure(background='#BBFFFF')
        # 初始化柱状图的颜色和标签的颜色
        self.colors = {
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
        # 创建组件
        self.initWidgets()
        # 设置组件布局
        self.setLayout()
        # 药品说明书源文件
        self.filename = None
        # 识别后的命名实体集合
        self.entities = None
        # 绘制空图占位
        self.drawBar(['CONTENT','REASON','TIME','DSPEC','MRL','SYMPTOM','ROA','PE',
                    'DRUG','FREQ','MEDICINE','SDOSE','ARL','IL','DISEASE','CROWD'],[10 for _ in range(16)])
        # 设置点击右上角叉号关闭程序
        self.window.protocol('WM_DELETE_WINDOW', self.on_closing)
        # 设置文件筐的选型
        self.file_opt = {
            'defaultextension':'.txt', # 默认文件类型
            'filetypes':[('all files', '.*'), ('text files', '.txt')] ,
            'initialdir':'datas/processed/', # 默认打开路径
        }
        # 识别内容
        self.content = []
        # self.option表示识别的内容是来自于文件还是输入，0为未选择、1为文件
        self.option = 0


    def initWidgets(self):
        """
        初始化窗口组件
        """
        # 设置文本显示框
        self.text_box = tk.Text(self.window,font=('微软雅黑', 9))
        # 设置选择文件按钮
        self.select_btn = tk.Button(self.window, text='选择文件', font=('微软雅黑', 14, 'bold'), 
                                    command=self.selectPath,bg='#FFFFFF',fg='#FF1493')
        # 设置开始识别按钮
        self.excute_btn = tk.Button(self.window, text='开始识别', font=('微软雅黑', 14, 'bold'), 
                                    command=self.startRecoginize,bg='#FFFFFF',fg='#FF1493')
        # 设置结果保存按钮
        self.save_btn = tk.Button(self.window, text='导出结果', font=('微软雅黑', 14, 'bold'),
                                     command=self.saveFile,bg='#FFFFFF',fg='#FF1493')

    def setLayout(self):
        """
        设置窗口的界面布局
        """
        # 设置文本框
        self.text_box.grid(row=0,column=0,rowspan=16,columnspan=10,padx=15,pady=20,sticky='e,w,s,n')
        # 设置文本框中的各种tag的样式
        self.text_box.tag_config('CONTENT',background=self.colors['CONTENT'])
        self.text_box.tag_config('REASON',background=self.colors['REASON'])
        self.text_box.tag_config('TIME',background=self.colors['TIME'])
        self.text_box.tag_config('DSPEC',background=self.colors['DSPEC'])
        self.text_box.tag_config('MRL',background=self.colors['MRL'])
        self.text_box.tag_config('SYMPTOM',background=self.colors['SYMPTOM'])
        self.text_box.tag_config('ROA',background=self.colors['ROA'])
        self.text_box.tag_config('PE',background=self.colors['PE'])
        self.text_box.tag_config('DRUG',background=self.colors['DRUG'])
        self.text_box.tag_config('FREQ',background=self.colors['FREQ'])
        self.text_box.tag_config('MEDICINE',background=self.colors['MEDICINE'])
        self.text_box.tag_config('SDOSE',background=self.colors['SDOSE'])
        self.text_box.tag_config('ARL',background=self.colors['ARL'])
        self.text_box.tag_config('IL',background=self.colors['IL'])
        self.text_box.tag_config('DISEASE',background=self.colors['DISEASE'])
        self.text_box.tag_config('CROWD',background=self.colors['CROWD'])
        # 设置按钮
        self.select_btn.grid(row=18,column=1,rowspan=3,columnspan=2,sticky='e,w,s,n')
        self.excute_btn.grid(row=18,column=4,rowspan=3,columnspan=2,sticky='e,w,s,n')
        self.save_btn.grid(row=18,column=7,rowspan=3,columnspan=2,sticky='e,w,s,n')


    def selectPath(self):
        try:
            self.option = 1
            self.text = []
            self.filename = askopenfilename(**self.file_opt)
            with open(self.filename,'r',encoding='utf-8') as fp:
                self.text_box.delete('1.0','end')
                # 获取每行文本的内容
                self.content = fp.readlines()
                self.text_box.insert(INSERT,''.join(self.content))
        except FileNotFoundError:
            messagebox.showinfo(title='tip',message='未选择文件！！！')

    def startRecoginize(self):
        """
        NER算法执行
        """
        if self.option == 0:
            receiver = self.text_box.get("1.0","end")
            self.content = []
            for line in receiver.split("\n"):
                self.content.append(line + "\n")
        if self.content == None:
            messagebox.showinfo(title='tip',message='未选择任何文件也未输入任何内容')
        else:
            # 预测实体
            self.entities = self.pred_model.predict(self.content)
            # 将识别的实体结果显示在窗口上
            self.text_box.delete('1.0','end')
            df = pd.DataFrame(self.entities)
            # 渲染识别成功的实体
            self.renderLabel(df)
            # 统计各类实体的数量
            nums,labels = groupByLabel(df)
            # 绘制实体图
            self.drawBar(labels,nums)
            # 重置输入选项
            self.option = 0

    def saveFile(self):
        """
        保存识别结果
        """
        try:
            save_path = asksaveasfilename(**self.file_opt)
        except FileNotFoundError:
            messagebox.showinfo(title='tip',message='未选择保存文件！！！')
        if self.entities != None:
            df = pd.DataFrame(self.entities)
            df.to_csv(save_path,index=False,encoding="utf-8")
            messagebox.showinfo(title='tip',message='结果成功保存')
        else:
            messagebox.showinfo(title='tip',message='没有识别到的实体可以保存！！！')

    def on_closing(self):
        if messagebox.askokcancel("Quit", "确认退出程序?"):
            self.window.destroy()
            sys.exit(0)

    def run(self):
        self.window.mainloop()

    def autolable(self,rects,gap=10):
        """
        绘制柱形图的值
        gap：文字距离柱形的距离
        """
        for rect in rects:
            height = rect.get_height()
            if height>=0:
                plt.text(rect.get_x()+rect.get_width()/2.0 - 0.16,height + gap,'{}'.format(height))
            else:
                plt.text(rect.get_x()+rect.get_width()/2.0 - 0.16,height - gap,'{}'.format(height))
                # 如果存在小于0的数值，则画0刻度横向直线
                plt.axhline(y=0,color='black')

    def drawBar(self,x,height):
        """
        绘制条形图
        gap：文字距离柱形的距离
        """
        fig = plt.figure(figsize=(5.2,5))
        plt.xticks(rotation = 50)
        plt.tick_params(axis='x', labelsize=7) 
        c = []
        for label in x:
            c.append(self.colors[label]) 
        ax = plt.bar(x,height,color=c)
        self.autolable(ax,gap=0.1)
        bar1 = FigureCanvasTkAgg(fig,self.window)
        bar1.get_tk_widget().grid(row=0,column=11,padx=5,rowspan=16,columnspan=16,pady=20,sticky='e,w,s,n')

    def renderLabel(self,df):
        """
        渲染标签
        df：dataframe 识别结果
        """
        entities = df.values.tolist()
        curlen = 0
        for i,line in enumerate(self.content):
            # line += '\n'
            a = str(i+1)+'.0'
            self.text_box.insert(a,line) #申明使用tag中的设置
            nextlen = curlen + len(line)
            for entity in entities:
                if entity[0] > curlen and entity[1] <= nextlen:
                    self.text_box.tag_add(entity[-1],'{}.{}'.format(i + 1, entity[0] - curlen), '{}.{}'.format(i + 1, entity[1] - curlen))
            curlen = nextlen



if __name__ == "__main__":
    NERSystem().run()


