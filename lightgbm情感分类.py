import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb

#把英文中的特殊符号去掉，只保留字母。
def preprocess_sentence(s):
    s=s.lower()
    s = re.sub(r"[^a-z]+", " ", s)
    s = s.rstrip().strip()
    return s
#把评论文本和标签存储为csv文件。    
def list_to_csv(contents,labels,path): #输入为文本列表和标签列表
    columns = ['contents', 'labels']
    save_file = pd.DataFrame(columns=columns, data=list(zip(contents, labels)))
    save_file.to_csv(path, index=False, encoding="utf-8")
#完整数据预处理方法
def prepare_csv(path,top,path2):
    labels=[]
    txts=[]
    i=1
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            result_dict = json.loads(line)
            s=preprocess_sentence(result_dict['text'])
            l=0
            if int(result_dict['stars'])>2 :
                l=1
            txts.append(s)
            labels.append(l)
            if i==top:
                break
            i+=1
    list_to_csv(txts,labels,path2)

if __name__ == '__main__':
    #第一次用完后，就注释掉数据预处理代码，使用其生成的csv文件即可。
    #prepare_csv('yelp_academic_dataset_review.json',10000,'yelp10000.csv')
    train_data = pd.read_csv('yelp10000.csv', sep=',', names=['contents', 'labels'],skiprows=1).astype(str)
    x_train, x_test, y_train, y_test = train_test_split(train_data['contents'], train_data['labels'], test_size=0.1)
    to_int = lambda x: int(x)
    x_train = x_train
    y_train = np.array(y_train.apply(to_int))
    x_test = x_test
    y_test = np.array(y_test.apply(to_int))
    # 将评论文本转化为词袋向量
    vectorizer = CountVectorizer(max_features=5000)
    tf_idf_transformer = TfidfTransformer()
    #根据词向量统计TF-IDF
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
    x_train_weight = tf_idf.toarray() 
    tf_idf = tf_idf_transformer.transform(vectorizer.transform(x_test))
    x_test_weight = tf_idf.toarray() 
    #构建lightgbm数据集
    lgb_train = lgb.Dataset(x_train_weight, y_train)
    lgb_val = lgb.Dataset(x_test_weight, y_test, reference=lgb_train)
    # 配置lightGBM参数
    params = {
    'max_depth': 12, 
    'num_leaves': 2048,
    'learning_rate': 0.05,
    'objective': 'multiclass', 
    'num_class': 2, 
    'verbose': -1
    }
    # 设置训练轮数
    num_boost_round = 1000
    # 开始训练
    gbm = lgb.train(params, lgb_train, num_boost_round, verbose_eval=100, valid_sets=lgb_val)
    # 预测数据集
    y_pred = gbm.predict(x_test_weight, num_iteration=gbm.best_iteration)
    y_predict = np.argmax(y_pred, axis=1)  # 获得最大概率对应的标签
    label_all = ['负面', '正面'] #设置标签名称
    confusion_mat = metrics.confusion_matrix(y_test, y_predict) 
    df = pd.DataFrame(confusion_mat, columns=label_all)
    df.index = label_all
    print('训练后，评估准确率为：', metrics.accuracy_score(y_test, y_predict))
    print('评估报告:', metrics.classification_report(y_test, y_predict))


