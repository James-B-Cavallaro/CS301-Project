```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
#import jax
#import jax.numpy as jnp

#columns = ['id', 'case_num', 'pn_num', 'feature_num', 'annotation', 'location']

data = pd.read_csv("https://raw.githubusercontent.com/James-B-Cavallaro/CS301-Project/main/train.csv")

print(data.head())
```

              id  case_num  pn_num  feature_num  \
    0  00016_000         0      16            0   
    1  00016_001         0      16            1   
    2  00016_002         0      16            2   
    3  00016_003         0      16            3   
    4  00016_004         0      16            4   
    
                                     annotation              location  
    0          ['dad with recent heart attcak']           ['696 724']  
    1             ['mom with "thyroid disease']           ['668 693']  
    2                        ['chest pressure']           ['203 217']  
    3      ['intermittent episodes', 'episode']  ['70 91', '176 183']  
    4  ['felt as if he were going to pass out']           ['222 258']  
    


```python

data['dictionary'] = data['annotation'].str.lower()
data['dictionary'] = data['annotation'].str.lstrip()
data['dictionary'] = data['annotation'].str.split()
print(data.head())
```

              id  case_num  pn_num  feature_num  \
    0  00016_000         0      16            0   
    1  00016_001         0      16            1   
    2  00016_002         0      16            2   
    3  00016_003         0      16            3   
    4  00016_004         0      16            4   
    
                                     annotation              location  \
    0          ['dad with recent heart attcak']           ['696 724']   
    1             ['mom with "thyroid disease']           ['668 693']   
    2                        ['chest pressure']           ['203 217']   
    3      ['intermittent episodes', 'episode']  ['70 91', '176 183']   
    4  ['felt as if he were going to pass out']           ['222 258']   
    
                                              dictionary  
    0             [['dad, with, recent, heart, attcak']]  
    1                 [['mom, with, "thyroid, disease']]  
    2                              [['chest, pressure']]  
    3           [['intermittent, episodes',, 'episode']]  
    4  [['felt, as, if, he, were, going, to, pass, ou...  
    


```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('dictionary')),
                          columns=mlb.classes_,
                          index=data.index))
```


```python
print(data.head())
```

              id  case_num  pn_num  feature_num  \
    0  00016_000         0      16            0   
    1  00016_001         0      16            1   
    2  00016_002         0      16            2   
    3  00016_003         0      16            3   
    4  00016_004         0      16            4   
    
                                     annotation              location  "Bleeding  \
    0          ['dad with recent heart attcak']           ['696 724']          0   
    1             ['mom with "thyroid disease']           ['668 693']          0   
    2                        ['chest pressure']           ['203 217']          0   
    3      ['intermittent episodes', 'episode']  ['70 91', '176 183']          0   
    4  ['felt as if he were going to pass out']           ['222 258']          0   
    
       "I  "Last  "Thyroid  ...  yo',  yo']  yr  yr']  yrs  yrs']  ~  ~3/4  ~8-10  \
    0   0      0         0  ...     0     0   0     0    0      0  0     0      0   
    1   0      0         0  ...     0     0   0     0    0      0  0     0      0   
    2   0      0         0  ...     0     0   0     0    0      0  0     0      0   
    3   0      0         0  ...     0     0   0     0    0      0  0     0      0   
    4   0      0         0  ...     0     0   0     0    0      0  0     0      0   
    
       ~q28d',  
    0        0  
    1        0  
    2        0  
    3        0  
    4        0  
    
    [5 rows x 5431 columns]
    


```python

data = data.drop(columns=['annotation'])
data = data.drop(columns=['location'])
data = data.drop(columns=['id'])
#data = data.drop(columns=['feature_num'])
print(data.head())
```

       case_num  pn_num  feature_num  "Bleeding  "I  "Last  "Thyroid  "Tums  \
    0         0      16            0          0   0      0         0      0   
    1         0      16            1          0   0      0         0      0   
    2         0      16            2          0   0      0         0      0   
    3         0      16            3          0   0      0         0      0   
    4         0      16            4          0   0      0         0      0   
    
       "associated  "bleeding  ...  yo',  yo']  yr  yr']  yrs  yrs']  ~  ~3/4  \
    0            0          0  ...     0     0   0     0    0      0  0     0   
    1            0          0  ...     0     0   0     0    0      0  0     0   
    2            0          0  ...     0     0   0     0    0      0  0     0   
    3            0          0  ...     0     0   0     0    0      0  0     0   
    4            0          0  ...     0     0   0     0    0      0  0     0   
    
       ~8-10  ~q28d',  
    0      0        0  
    1      0        0  
    2      0        0  
    3      0        0  
    4      0        0  
    
    [5 rows x 5428 columns]
    


```python
numcol = len(data.columns)

y = data['feature_num']
data = data.drop(columns=['feature_num'])
X = data.iloc[:, 0:numcol].values
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```


```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, max_depth=1000, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```


```python
from sklearn import metrics
predictions = classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
```

    Accuracy: 0.6397202797202797
    


```python
import matplotlib.pyplot as plt

jaXarr = jnp.array(X)
jaxYarr = jnp.array(y)

plt.title("Train vs Target")
plt.xlabel("Train")
plt.ylabel("Target")
plt.plot(jaXarr,jaxYarr)
plt.show()
```
