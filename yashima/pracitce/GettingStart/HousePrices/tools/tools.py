import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Core():

  def __init__(self, x_train, x_test, y_train):
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.x_all = pd.concat([self.x_train, self.x_test], sort=False, ignore_index=True, axis=0)

    self.n_train = len(self.x_train)
    self.n_test = len(self.x_test)
    self.n_all = len(self.x_all)

    self.col_object = self.x_train.select_dtypes(include='object')
    self.col_int = self.x_train.select_dtypes(include='int')
    self.col_float = self.x_train.select_dtypes(include='float')
    self.col_numeric = self.x_train.select_dtypes(exclude='object')

    self.missing = self.x_all.isnull().sum()
    
  def loadDF(self, x_train, x_test, y_train):
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.x_all = pd.concat([self.x_train, self.x_test], sort=False, ignore_index=True, axis=0)

    self.n_train = len(self.x_train)
    self.n_test = len(self.x_test)
    self.n_all = len(self.x_all)

    self.col_object = self.x_train.select_dtypes(include='object')
    self.col_int = self.x_train.select_dtypes(include='int')
    self.col_float = self.x_train.select_dtypes(include='float')
    self.col_numeric = self.x_train.select_dtypes(exclude='object')
    
    self.missing = self.x_all.isnull().sum()

  def split(self):
    self.x_train = self.x_all.iloc[self.n_train:].reset_index()
    self.x_test = self.x_all.iloc[:self.n_train].reset_index()


class Preprocess(Core):
  
  def __init__(self, x_train, x_test, y_train):
    super().__init__(x_train, x_test, y_train)


  def pull(self):
    return self.x_train, self.x_test


  def fill(self, fill_method_num='mean', fill_method_object='Missing', help=False):
    method_num = ('mean', 'median', 'mode', 0)
    method_object = ('mode', 'Missing')

    if help:
      print('''
      method_num must be in {}. \n
      method_object must be in {}.
      '''.format(method_num, method_object))
      return

    if fill_method_num not in method_num:
      print('''
      Error! fill_method_num must be in {}.
      '''.format(method_num)) 
      return
    if fill_method_object not in method_object:
      print('''
      Error! fill_method_object must be in {}.
      '''.format(fill_method_object))

    missing = self.missing[self.missing>0]
    for col in list(missing.index):

      if self.x_all[col].dtype=='int64' or 'float64' or 'int16' or 'float16':
        if fill_method_num is 'mean':
          self.x_all[col].fillna(self.x_all[col].mean(), inplace=True)
        elif fill_method_num is 'median':
          self.x_all[col].fillna(self.x_all[col].median(), inplace=True)
        elif fill_method_num is 'mode':
          self.x_all[col].fillna(self.x_all[col].mode(), inplace=True)
        elif fill_method_num is 0:
          self.x_all[col].fillna(0, inplace=True)

      if self.x_all[col].dtype=='object':
        if fill_method_object is 'mode':
          self.x_all[col].fillna(self.x_all[col].value_counts().index[0], inplace=True)
        if fill_method_object is 'Missing':
          self.x_all[col].fillna('Missing', inplace=True)

    Core.split(self)

    print('Missing NAN of Numeric types is filled by : {}.'.format(fill_method_num))
    print('Missing NAN of Numeric types is filled by : {}.'.format(fill_method_object))


  def encode(self, method_enc='label'):
    col_object = self.col_object
    col_object_ind = []
    for col in col_object:
      col_object_ind.append(self.x_all.columns.get_loc(col))
    if method_enc is 'label':
      label_enc = LabelEncoder()
      for i in col_object_ind:
        self.x_all.iloc[:,i] = label_enc.fit_transform(self.x_all.iloc[:,i])
    
    Core.split(self)


  def count_byChi2(self, col, significance):
    data = self.x_all.fillna(self.x_all.median())
    data = data[col]
    data = data.reset_index(drop=True)
    mean = data.mean()
    variance = data.var()
    anomaly_scores = []
    for x in data:
      anomaly_score = (x-mean)**2/variance
      anomaly_scores.append(anomaly_score)
    threshold = stats.chi2.interval(1-significance, 1)[1]
    outliers = []
    for i, an in enumerate(anomaly_scores):
      if an > threshold:
        outliers.append(i)
    return outliers


  def view(self, dtype='all', significance=0.01, help=False):
    dtypes = (
              # 'all', 
              'numeric', 'int', 'float',
              # 'object',
              'custom'
              )
    if help:
      print('''
      dtpye must be in {}.
      '''.format(dtypes))
      return
    if dtype not in dtypes:
      print('''
      Error! dtpye must be in {}.
      '''.format(dtypes))
      return

    if dtype is 'all':
      columns = self.x_all.columns
    elif dtype is 'numeric':
      columns = self.col_numeric
    elif dtype is 'int':
      columns = self.col_int
    elif dtype is 'float':
      columns = self.col_float
    elif dtype is 'object':
      columns = self.col_object
    if dtype is 'custom':
      columns = [col for col in self.x_all.columns if col in self.col_numeric]

    colorlist = ["r", "g", "b", "c", "m", "y"]
    for i, col in enumerate(columns):       
      outliers = self.count_byChi2(col, significance)
      
      self.x_all[col].plot(figsize=(7,1.5), color=colorlist[i%len(colorlist)])
      if len(outliers) is not 0:
        out_min = pd.Series(np.zeros(self.x_all[col].shape))
        out_min[:] = min(self.x_all.reset_index(drop=True)[col][outliers])
        out_min.plot(figsize=(7,1.5), color='black')
        # out_max = pd.Series(np.zeros(self.x_all[col].shape))
        # out_max[:] = max(self.x_all.reset_index(drop=True)[col][outliers])
        # out_max.plot(figsize=(7,1.5), color='black')
      plt.show()
        
      plt.figure(figsize=(7,1.5))
      sns.distplot(self.x_all[col].dropna(), hist=True, rug=True,
                   color=colorlist[i%len(colorlist)])
      plt.show()
        
      sns.boxplot(data=self.x_all[col].dropna())
      plt.show()
        
      print ('{} has {} NaNs ({:.2f}%).'.format(col, self.x_all[col].isnull().sum(), self.x_all[col].isnull().sum()/self.n_all*100))
      print('Skewness : ', self.x_all[col].skew())
      print('Kurtosis : ', self.x_all[col].kurt())
      print('Number of anomaly scores over threshold({}%) : {} / {}'
            .format(significance*100, len(outliers), len(self.x_all[col])))
      if len(outliers) is not 0:
        print('Border line : ', out_min[0])
      else:
        pass
      plt.close()
      print('-'*100)
      
      
  def NANs(self, top=10, bar=True, plot=False, get_return=False):
    col_NANs = self.x_all.isnull().sum().sort_values()
    col_NANs = col_NANs[len(col_NANs)-top:]
    col_NANs = col_NANs/self.n_all
    if bar:
      col_NANs.plot.bar(figsize=(7,3))
      plt.show()
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[col_NANs.index]
      self.view('custom',significance=0.01)
      self.x_all = tmp.copy()   
    if get_return:
      return col_NANs
    
  def Skews(self, top=10, bar=True, plot=False,get_return=False):
    col_Skews = self.x_all.skew().sort_values()
    col_Skews = col_Skews.iloc[len(col_Skews)-top:]
    if bar:
      col_Skews.plot.bar(figsize=(7,3))
      plt.show()
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[col_Skews.index]
      self.view('custom',significance=0.01)
      self.x_all = tmp.copy()   
    if get_return:
      return col_Skews
    
    
  def Kurts(self, top=10, bar=True, plot=False, get_return=False):
    col_Kurts = self.x_all.kurt().sort_values()
    col_Kurts = col_Kurts.iloc[len(col_Kurts)-top:]
    if bar:
      col_Kurts.plot.bar(figsize=(7,3))
      plt.show()
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[col_Kurts.index]
      self.view('custom',significance=0.01)
      self.x_all = tmp.copy()   
    if get_return:
      return col_Kurts