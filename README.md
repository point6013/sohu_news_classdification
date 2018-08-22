

```python
import numpy as np
import pandas as pd
```


```python
df=pd.read_excel('sohu_news.xlsx')
df['length']=df['content'].apply(lambda x: len(x)).values
```


```python
df_data = df[df['length']>=50][['content','category']]
```


```python
df_data['category'].value_counts()
# 可以看到这里面存在类别不平衡，最大的差距有7倍。
```




    health      30929
    news        27613
    auto        22841
    stock       18152
    it          13875
    yule        13785
    women        4667
    book         4411
    business     1769
    Name: category, dtype: int64




```python
from sklearn.preprocessing import LabelEncoder
class_le=LabelEncoder()
y=class_le.fit_transform(df['category'].values)
y[:20]
```




    array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
          dtype=int64)




```python
import jieba
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
X=pd.DataFrame()
X['cut_content']=df["content"].apply(chinese_word_cut)
X['cut_content'].head()
```

    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\FRED-H~1\AppData\Local\Temp\jieba.cache
    Loading model cost 0.600 seconds.
    Prefix dict has been built succesfully.
    




    1    产品名称 ：  规格 及 价格 ： ３ ０ ｍ ｌ ／ ３ ０ ０ 　 元  羽西 当归...
    2    常见问题  Ｑ ： 为什么 我 提交 不了 试用 申请 　 Ａ ： 试用 申请 必须 同时...
    3    产品名称 ： 肌醇 （ Ｐ ｕ ｒ ｅ 　 Ｓ ｋ ｉ ｎ ） 深层 卸妆 凝胶  规格 ...
    4    欧诗漫 的 试用装 终于 延期 而 至 ， 果然 不负 所望 ， 包装 很 精美 。 从 快...
    5    试用 申请 步骤  １ 注册 并 完善 个人资料 　 登入 搜狐 试用 频道 ， 填写 并...
    Name: cut_content, dtype: object




```python
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=42,test_size=0.25)
def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,encoding="utf-8") as f:
        custom_stopwords_list=[i.strip() for i in f.readlines()]
    return custom_stopwords_list
stop_words_file = "stopwords.txt"
stopwords = get_custom_stopwords(stop_words_file) # 获取停用词
from sklearn.feature_extraction.text import  CountVectorizer
vect = CountVectorizer(token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',stop_words=frozenset(stopwords))
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
from sklearn.pipeline import make_pipeline
pipe=make_pipeline(vect,nb)
pipe.fit(X_train.cut_content, y_train)
y_pred = pipe.predict(X_test.cut_content)
from sklearn import  metrics
print(metrics.accuracy_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)
```

    0.8972162169375717
    




    array([[6266,  163,    2,  249,    5,  345,   66,   74,   53],
           [   5, 1118,    0,    0,    0,   31,    2,    5,   37],
           [   8,    4,   15,    0,    0,  104,  329,    5,    3],
           [   4,    1,    0, 8230,    0,   64,    6,    1,    0],
           [  59,   29,    0,   10, 3672,   66,   29,   26,   45],
           [  72,   71,    6,   26,    1, 5683,  756,   60,  193],
           [  28,    0,   10,    0,    0,  381, 4275,    0,    2],
           [   9,   90,    0,    5,    1,   74,    5,  890,  132],
           [   2,   38,    1,    2,    0,   44,    1,   11, 3467]],
          dtype=int64)




```python
df['category'].value_counts()
```




    health      33044
    auto        28696
    news        27815
    stock       18918
    it          15589
    yule        14435
    women        4790
    book         4724
    business     1856
    Name: category, dtype: int64




```python
y_pred = pipe.predict(X_train.cut_content)
from sklearn import  metrics
print(metrics.accuracy_score(y_train,y_pred))
```

    0.9131583629893238
    


```python
from sklearn.linear_model import LogisticRegression
```


```python
lr=LogisticRegression()  
from sklearn.pipeline import make_pipeline
pipe=make_pipeline(vect,lr)
pipe.fit(X_train.cut_content, y_train)
y_pred = pipe.predict(X_test.cut_content)
from sklearn import  metrics
print(metrics.accuracy_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)
```


```python
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='gini',n_estimators=30,random_state=1,n_jobs=2)
pipe=make_pipeline(vect,forest)
pipe.fit(X_train.cut_content, y_train)
y_pred = pipe.predict(X_test.cut_content)
print(metrics.accuracy_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)
```


```python
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',random_state=1)
from sklearn.ensemble import BaggingClassifier
bag=BaggingClassifier(base_estimator=tree,
                     n_estimators=10,
                     max_samples=1.0,
                     max_features=1.0,
                     bootstrap=True,
                     bootstrap_features=False,
                     n_jobs=4,random_state=1)
pipe=make_pipeline(vect,bag)
pipe.fit(X_train.cut_content, y_train)
y_pred = pipe.predict(X_test.cut_content)
metrics.accuracy_score(y_test,y_pred)
```




    0.9294045426642111


