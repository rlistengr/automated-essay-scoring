import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
import word2vec
from math import ceil 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import string


# 读取数据
data = pd.read_csv('../data/essays_and_scores.csv', encoding = 'latin-1')

# 利用boxplot来可视化domain1_score的分布。 
%matplotlib inline
data.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10, 10))


# 特征工程
def get_embedding(essay):
    """获取文本的词向量特征
    词向量已经使用essay在本地训练完毕，没有上传相关的训练过程，因为调用word2vec包较为简单
    通过采样句子中的单词的方式获取文本的词向量特征
        每个200个单词猜一次样，采500个样，不够的话使用循环的方式进行补齐
    Args：
        文章
    Returns：
        文章的词向量特征
    """
    model = word2vec.load('../data/essay.bin')
    result = []
    wordnotfind = [0]*100
    for article in essay:
        cycle = ceil(len(article) / 200)        
        # 获取单词个数，如果长度小于500，则通过循环复制补长到500
        if len(article) < 500*cycle:
            article = article * ceil(500 / len(article))
        
        
        # 每间隔cycle个单词采一次样
        tmp = []
        for i in range(200):
            try:
                wordv = model[article[i*cycle]]
            except:
                wordv = wordnotfind
            tmp.append(wordv)
            
        # tmp = [model[article[i*cycle]] for i in range(500)]

        # 拼接成一个行向量
        vector = np.array(tmp).reshape(1,-1).tolist()[0]

        # 添加在最后一行
        result.append(vector)
            
    return result
        
def avg_word_len(essay):
    """
    文章中单词的平均长度
    """
    return [len(''.join(article)) / len(article) for article in essay]  
    
def word_count(essay):
    """
    文章中单词的总数
    """
    return [len(article) for article in essay]
    
def setence_count(essay):
    """
    文章中句子的个数
    """
    return [len(sent_tokenize(article)) for article in essay]
    
def count_spell_error(essay):
    """
    文章中有多少个单词的拼错的,需要用到词库（words.txt）
    """
    result = []
    stopwords_ = stopwords.words('english')
    
    with open('../data/words.txt', encoding='utf8') as f:
        vocabulary = f.read()
        vocabulary = vocabulary.split()
    
    for article in essay:
        num = 0
        for word in article:
            if word not in vocabulary and word not in stopwords_:
                num += 1
        result.append(num)
    
    return result

def count_pos(essay):
    """
    出现了多少个名词，动词以及形容词
    """
    result = []
    
    for article in essay:
        a1 = len([0 for pos in nltk.pos_tag(article) if pos[1][:2] == 'NN'])
        a2 = len([0 for pos in nltk.pos_tag(article) if pos[1][:2] == 'VB'])
        a3 = len([0 for pos in nltk.pos_tag(article) if pos[1][:2] == 'JJ'])
        
        result.append([a1,a2,a3])
    return result

def extract_features(data):
    """提取数据特征
    通过前面定义的特征工程函数，依次提取各个特征，并拼接在一起
    Args:
        文章集合
    Returns：
        转化为特征向量的文章矩阵
    """
    essay = []
    for article in data:
        essay.append([word for word in nltk.word_tokenize(re.sub('[%s]' % re.escape(string.punctuation),' ', article))])
    
    # 获得词向量矩阵
    features = np.array(get_embedding(essay))
    
    # 获取单词平均长度的列向量，与前面矩阵拼接
    features = np.c_[features, avg_word_len(essay)]

    # 获取单词总数的列向量，与前面矩阵拼接
    features = np.c_[features, word_count(essay)]   

    # 获取句子个数，与前面矩阵拼接
    features = np.c_[features, setence_count(data)]   
    
    # 获取各词性单词个数矩阵，与前面的矩阵拼接
    features = np.c_[features, count_pos(essay)]  

    return features

    
features = extract_features(data.loc[:, 'essay'])
X = features.tolist()
y = data.loc[:,'domain1_score'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

'''
利用GBDT模型做预测
'''

params = {'n_estimators':[100], 'max_depth':[5,10], 'min_samples_split':[4], 'learning_rate':[0.1], 'loss':['ls']}
gbr = ensemble.GradientBoostingRegressor()

'''
# 使用BayesianOptimization 搜索参数

from bayes_opt import BayesianOptimization

def score_function(n_estimators_, max_depth_, min_samples_split_, learning_rate_):
    gbr_ = ensemble.GradientBoostingRegressor(n_estimators=int(n_estimators_), 
                                              max_depth=int(max_depth_), 
                                              min_samples_split=int(min_samples_split_),
                                              learning_rate = learning_rate_,
                                              loss = 'ls')
    gbr_.fit(X_train, y_train)

    return gbr_.score(X_train, y_train)

optimizer = BayesianOptimization(
            score_function,
            {'n_estimators_': (100,1000),
             'max_depth_': (5,30),
             'min_samples_split_':(3,6),
             'learning_rate_':(0.1,0.2)})

optimizer.maximize(
    init_points=3,
    n_iter=2,
)

grid = ensemble.GradientBoostingRegressor(n_estimators=int(optimizer.max['params']['n_estimators_']),
                                          max_depth=int(optimizer.max['params']['max_depth_']),
                                          min_samples_split=int(optimizer.max['params']['min_samples_split_']),
                                          learning_rate=optimizer.max['params']['learning_rate_'],
                                          loss='ls')
grid.fit(X_train, y_train)
'''

grid = GridSearchCV(gbr, params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.alpha)


# 模型评估
y_pred = grid.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE: %.4f" % mse)

print('Variance score: %.2f' % grid.score(X_test, y_test))

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))