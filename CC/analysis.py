import os
from ICCSupervised.ICCSupervised import IAnalysis
import json

class CCAnalysis(IAnalysis):

    def __init__(self, tag2idx, idx2tag):
        self.train_record = {
            'loss': [],
            'f1': [],
            'acc': [],
            'recall': [],
            'reports' : []
        }


        self.eval_record = {
            'loss': [],
            'f1': [],
            'acc': [],
            'recall': [],
            'reports': []
        }


        self.model_record = []

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
    
    def append_train_record(self, train_record_item):
        for key in train_record_item:
            self.train_record[key].append(train_record_item[key])
    
    def append_eval_record(self, eval_record_item):
        for key in eval_record_item:
            self.eval_record[key].append(eval_record_item[key])
    
    def append_model_record(self, uid):
        self.model_record.append(uid)
    
    def getPrecision(self, tags, pred):
        pred_num = 0
        correct_num = 0
        for i, item in enumerate(pred):
            t_tags = tags[i].tolist()
            insider = False
            eq_tag = False
            for j, tag in enumerate(item):
                if tag != self.tag2idx('O'):
                    if insider == False:
                        pred_num += 1
                        insider = True
                        eq_tag = True
                    if tag != t_tags[j]:
                        eq_tag = False
                else:
                    insider = False
                    if eq_tag == True:
                        correct_num += 1
                    eq_tag = False
        return pred_num, correct_num


    def getRecall(self, tags):
        gold_num = 0
        for i, item in enumerate(tags):
            t_tags = item.tolist()
            for tag in t_tags:
                if self.idx2tag(tag).find('B-') > -1:
                    gold_num += 1
                elif self.idx2tag(tag).find('S-') > -1:
                    gold_num += 1
        return gold_num
    
    def save_ner_record(self, uid):
        path = f'./data_record/{uid}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/train.csv', encoding='utf-8', mode='w+') as f:
            result = 'F1,Acc,Recall,loss\n'
            for i, item in enumerate(self.train_record['loss']):
                result += '{},{},{},{}\n'.format(self.train_record['f1'][i], self.train_record['acc'][i], self.train_record['recall'][i], self.train_record['loss'][i])
            f.write(result)
        
        if not os.path.exists(f"{path}/train_reports"):
            os.makedirs(f"{path}/train_reports")

        # save reports
        for i,r in enumerate(self.train_record['reports']):
            with open(f"{path}/train_reports/{i+1}_epoch.json",encoding='utf-8',mode="w") as f:
                f.write(json.dumps(r,ensure_ascii=False,indent=2))
        
        with open(f'{path}/eval.csv', encoding='utf-8', mode='w+') as f:
            result = 'F1,Acc,Recall,loss\n'
            for i, item in enumerate(self.eval_record['loss']):
                result += '{},{},{},{}\n'.format(self.eval_record['f1'][i], self.eval_record['acc'][i], self.eval_record['recall'][i], self.eval_record['loss'][i])
            f.write(result)

        if not os.path.exists(f"{path}/eval_reports"):
            os.makedirs(f"{path}/eval_reports")

        for i,r in enumerate(self.eval_record['reports']):
            with open(f"{path}/eval_reports/{i+1}_epoch.json",encoding='utf-8',mode="w") as f:
                f.write(json.dumps(r,ensure_ascii=False,indent=2))
        
        with open(f'{path}/model.csv', encoding='utf-8', mode='w+') as f:
            result = 'model_uid\n'
            for i, item in enumerate(self.model_record):
                result += '{}\n'.format(item)
            f.write(result)
        return uid
    
    @staticmethod
    def save_csv(dir, file_name, **args):
        if not os.path.isdir(dir):
            os.makedirs(dir)
        result = ''
        dicts = []
        for key in args.keys():
            dicts.append(key)
            result = key if result == '' else result + '\t{}'.format(key)
        length = len(args[dicts[0]])
        result += '\n'
        for i in range(length):
            t = True
            for key in args.keys():
                result += str(args[key][i]) if t else '\t{}'.format(args[key][i])
                t = False
            result += '\n'
        with open('{}/{}.csv'.format(dir, file_name), encoding='utf-8', mode='w+') as f:
            f.write(result)
