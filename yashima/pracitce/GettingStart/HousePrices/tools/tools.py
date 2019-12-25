import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy import special
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

    self.col_object = self.x_train.select_dtypes(include='object').columns
    self.col_int = self.x_train.select_dtypes(include='int').columns
    self.col_float = self.x_train.select_dtypes(include='float').columns
    self.col_numeric = self.x_train.select_dtypes(exclude='object').columns

    self.missing = self.x_all.isnull().sum()
    
  def loadDF(self, x_train, x_test, y_train):
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.x_all = pd.concat([self.x_train, self.x_test], sort=False, ignore_index=True, axis=0)

    self.n_train = len(self.x_train)
    self.n_test = len(self.x_test)
    self.n_all = len(self.x_all)

    self.col_object = self.x_train.select_dtypes(include='object').columns
    self.col_int = self.x_train.select_dtypes(include='int').columns
    self.col_float = self.x_train.select_dtypes(include='float').columns
    self.col_numeric = self.x_train.select_dtypes(exclude='object').columns
    
    self.missing = self.x_all.isnull().sum()

  def split(self):
    self.x_train = self.x_all.iloc[:self.n_train].reset_index(drop=True)
    self.x_test = self.x_all.iloc[self.n_train:].reset_index(drop=True)    
    
  def update(self):
    self.x_all = pd.concat([self.x_train, self.x_test], sort=False, ignore_index=True, axis=0)
    
    self.n_train = len(self.x_train)
    self.n_test = len(self.x_test)
    self.n_all = len(self.x_all)

    self.col_object = self.x_train.select_dtypes(include='object')
    self.col_int = self.x_train.select_dtypes(include='int')
    self.col_float = self.x_train.select_dtypes(include='float')
    self.col_numeric = self.x_train.select_dtypes(exclude='object')

    self.missing = self.x_all.isnull().sum()


class Process(Core):
  
  def __init__(self, x_train, x_test, y_train):
    super().__init__(x_train, x_test, y_train)

  '''
  INTERFACE
  '''
  def pull(self):
    return self.x_train, self.x_test, self.y_train



  '''
  PREPROCESS
  '''
  def fill(self, fill_method_num='mean', fill_method_object='None', help=False, inplace=False, get_return=False):
    df = self.x_all.copy()
    method_num = ('mean', 'median', 'mode', 0)
    method_object = ('mode', 'Missing', 'None')

    if help:
      print('''
      Usage:
      'method_num' must be in {}. 
      'method_object' must be in {}. 
      'inplace' decides update this object or not. 
      'get_return' decides return filed train and test data or not.
      '''.format(method_num, method_object))
      return 

    if fill_method_num not in method_num:
      print('''
      Error! 'fill_method_num' must be in {}.
      '''.format(method_num)) 
      return
    if fill_method_object not in method_object:
      print('''
      Error! 'fill_method_object must' be in {}.
      '''.format(fill_method_object))
      return

    missing = self.missing[self.missing>0]
    for col in list(missing.index):
      if df[col].dtype!='object':
        if fill_method_num is 'mean':
          df[col].fillna(df[col].mean(), inplace=True)
        elif fill_method_num is 'median':
          df[col].fillna(df[col].median(), inplace=True)
        elif fill_method_num is 'mode':
          df[col].fillna(df[col].mode(), inplace=True)
        elif fill_method_num is 0:
          df[col].fillna(0, inplace=True)

      if df[col].dtype=='object':
        if fill_method_object is 'mode':
          df[col].fillna(df[col].value_counts().index[0], inplace=True)
        if fill_method_object is 'Missing':
          df[col].fillna('Missing', inplace=True)
        if fill_method_object is 'None':
          df[col].fillna('None', inplace=True)
          
    if inplace:
      self.x_all = df.copy()
      Core.split(self)
      Core.update(self)

    print('Missing NAN of Numeric types is filled by : {}.'.format(fill_method_num))
    print('Missing NAN of Numeric types is filled by : {}.'.format(fill_method_object))
    
    if get_return:
      return df.iloc[:self.n_train].reset_index(drop=True), df.iloc[self.n_train:].reset_index(drop=True), self.y_train



  def encode(self, method_enc='label', inplace=False, get_return=True):
    df = self.x_all.copy()
    col_object = self.col_object
    col_object_ind = []
    for col in col_object:
      col_object_ind.append(df.columns.get_loc(col))
    if method_enc is 'label':
      label_enc = LabelEncoder()
      for i in col_object_ind:
        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])
    
    if inplace:
      self.x_all = df.copy()
      Core.split(self)
      Core.update(self)
      
    if get_return:
      return df.iloc[:self.n_train].reset_index(drop=True), df.iloc[self.n_train:].reset_index(drop=True), self.y_train



  def dropCorr(self, threshold, inplace=False, get_return=False):
    df = self.x_all.copy()
    df_corr = df.corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape),k=1).astype(np.bool))
    cols_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(cols_to_drop, axis=1)
    print('Droped columns by Corr : {}\n'.format(cols_to_drop))
    
    if inplace:
      self.x_all = df.copy()
      Core.split(self)
      Core.update(self)
      
    if get_return:
      return df.iloc[:self.n_train].reset_index(drop=True), df.iloc[self.n_train:].reset_index(drop=True), self.y_train
   
    

  def transY(self, method='log',inplace=False, get_return=False, help=False):
    methods = (
              'log',
              'boxcox'
              )
    if help:
      print('''
      Usage:
      'method' must be in {}.
      '''.format(methods))
      return
    if method not in methods:
      print('''
      Error! 'method' must be in {}.
      '''.format(methods))
      return
      
    if method is 'log':
      y_trans = np.log1p(self.y_train)
    elif method is 'boxcox':
      col = self.y_train.name
      y_trans, _ = stats.boxcox(self.y_train)
      y_trans = pd.Series(y_trans)
      y_trans.name = col
    
    if inplace:
      self.y_train = y_trans.copy()
      
    if get_return:
      return self.x_train, self.x_test, y_trans
 
  
  
  def transF(self, threshold=0.5, method='log',inplace=False, get_return=False, help=False):
    methods = (
              'log',
              'boxcox',
              # 'johnson'
              )
    if help:
      print('''
      Usage:
      'method' must be in {}.
      '''.format(methods))
      return
    if method not in methods:
      print('''
      Error! 'method' must be in {}.
      '''.format(methods))
      return
    
    df = self.x_all.copy()   
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = df[numeric_feats].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > threshold]
    skewed_features = high_skew.index
       
    if method is 'log':
      for feat in skewed_features:
        df[feat] = np.log1p(df[feat])      
    elif method is 'boxcox':
      for feat in skewed_features:
        df[feat] = special.boxcox1p(df[feat], stats.boxcox_normmax(df[feat] + 1))
    # elif method is 'johnson':
    #   for col in self.col_numeric:
    #     df[col], _ = stats.yeojohnson(df[col])
    
    if inplace:
      self.x_all = df.copy()
      Core.split(self)
      Core.update(self)
      
    if get_return:
      return df.iloc[:self.n_train].reset_index(drop=True), df.iloc[self.n_train:].reset_index(drop=True), self.y_train
  
    
    
  def dropVar(self, threshold=0, view=True, inplace=False, get_return=False):
    df = self.x_all.copy()
    sel = VarianceThreshold(threshold)
    sel.fit(df)
    cols_sel = sel.get_support()
    cols_drop = [col for i,col in enumerate(self.x_all.columns) if not cols_sel[i]]
    df = df.loc[:,cols_sel]
    
    if view:
      print('Drop columns : ',cols_drop)
      print('Droped {} columns.'.format(len(cols_drop)))
      
    if inplace:
      self.x_all = df.copy()
      Core.split(self)
      Core.update(self)
      
    if get_return:
      return df.iloc[:self.n_train].reset_index(drop=True), df.iloc[self.n_train:].reset_index(drop=True), self.y_train
    
    
    
  def rejectOut(self, columns, significance=0.01, inplace=False, get_return=False):
    df_x = self.x_train.copy()
    df_y = self.y_train.copy()
    count = 0
    for col in columns:
      outliers = self.count_byChi2(col, significance, self.x_all)
      for i in outliers:
        if i in df_x[col].index.to_list():
          df_x = df_x.drop(i, axis=0)
          df_y = df_y.drop(i, axis=0)
          count += 1
    print('Rejected data by Outliers from : ', columns)
    print('Rejected {} data.'.format(count))
    
    if inplace:
      self.x_train = df_x.copy()
      self.y_train = df_y.copy()
      self.x_all = pd.concat([self.x_train, self.x_test], sort=False, ignore_index=True, axis=0)
      Core.update(self)
      
    if get_return:
      return df_x, self.x_test, df_y

  '''
  ANALYSIS
  '''
  def count_byChi2(self, col, significance, df):
    data = df.fillna(df.median())
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



  def viewY(self, dtype='numeric'): 
    col = self.y_train.name    
    self.y_train.plot(figsize=(7,1.5), color='b')
    
    fig = plt.figure(figsize = (15,3))
    ax1 = fig.add_subplot(1, 2, 1)
    sns.distplot(self.y_train.dropna(), hist=True, rug=True,
                 color='b', ax=ax1)
      
    ax2 = fig.add_subplot(1, 2, 2)
    sns.boxplot(data=self.y_train.dropna(), ax=ax2)
    plt.show()
          
    print ('{} has {} NaNs ({:.2f}%).'.format(col, self.y_train.isnull().sum(), self.y_train.isnull().sum()/self.n_all*100))
    print('Skewness : {:.2f}'.format(self.y_train.skew()))
    print('Kurtosis : {:.2f}'.format(self.y_train.kurt()))
    plt.close()
    print('-'*130)



  def viewF(self, dtype='numeric', data='all', significance=0.01, help=False, input_cols=[]):
    dtypes = (
              # 'all', 
              'numeric', 'int', 'float',
              # 'object',
              'custom'
              )
    data_set = (
              'all', 
              'train',
              'test'
              )
    
    if help:
      print('''
      Usage:
      'dtpye' must be in {}.
      '''.format(dtypes))
      return
    if dtype not in dtypes:
      print('''
      Error! 'dtpye' must be in {}.
      '''.format(dtypes))
      return
    if data not in data_set:
      print('''
      Error! 'dtpye' must be in {}.
      '''.format(data_set))
      return
    
    if data is 'all':
      df = self.x_all.copy()
    if data is 'train':
      df = self.x_train.copy()
    if data is 'test':
      df = self.x_test.copy()

    if dtype is 'all':
      columns = self.df.columns
    elif dtype is 'numeric':
      columns = self.col_numeric
    elif dtype is 'int':
      columns = self.col_int
    elif dtype is 'float':
      columns = self.col_float
    elif dtype is 'object':
      columns = self.col_object
    if dtype is 'custom':
      columns = [col for col in df.columns if col in self.col_numeric]

    if len(input_cols) != 0:
      columns = input_cols

    colorlist = ["r", "g", "c", "m", "y"]
    for i, col in enumerate(columns):       
      outliers = self.count_byChi2(col, significance, df)
      
      df[col].plot(figsize=(7,1.5), color=colorlist[i%len(colorlist)])
      if len(outliers) is not 0:
        out_min = pd.Series(np.zeros(df[col].shape))
        out_min[:] = min(df.reset_index(drop=True)[col][outliers])
        out_min.plot(figsize=(7,1.5), color='black')
        # out_max = pd.Series(np.zeros(self.x_all[col].shape))
        # out_max[:] = max(self.x_all.reset_index(drop=True)[col][outliers])
        # out_max.plot(figsize=(7,1.5), color='black')
      plt.show()
      
      fig = plt.figure(figsize = (15,7))
      ax1 = fig.add_subplot(2, 2, 1)
      sns.distplot(df[col].dropna(), hist=True, rug=True,
                   color=colorlist[i%len(colorlist)], ax=ax1)

      ax2 = fig.add_subplot(2, 2, 2)
      sns.boxplot(data=df[col].dropna(), ax=ax2)
      
      df_tmp = pd.DataFrame({col:self.x_train[col],self.y_train.name:self.y_train})
      corr = df_tmp.corr()[col][1]
      ax3 = fig.add_subplot(2, 2, 3)
      sns.regplot(x = self.x_train[col], y = self.y_train, ax=ax3)
      ax4 = fig.add_subplot(2, 2, 4)
      sns.residplot(x = self.x_train[col], y = self.y_train, ax=ax4)
      plt.show()
          
      print ('{} has {} NaNs ({:.2f}%).'.format(col, df[col].isnull().sum(), df[col].isnull().sum()/self.n_all*100))
      print('Correlation Coefficient ({} vs {}): {:.3f}'.format(col, self.y_train.name, corr))
      print('Skewness : {:.2f}'.format(df[col].skew()))
      print('Kurtosis : {:.2f}'.format(df[col].kurt()))
      print('Number of anomaly scores over threshold({}%) : {} / {}'
            .format(significance*100, len(outliers), len(df[col])))
      if len(outliers) is not 0:
        print('Border line : ', out_min[0])
      else:
        pass
      plt.close()
      print('-'*130)
      
 
      
  def NANs(self, top=-1, bar=True, plot=False, get_return=False):
    col_NANs = self.x_all.isnull().sum().sort_values()
    if top==-1:
      len_NANs = len([col for col in col_NANs.index if col_NANs[col] != 0])
      col_NANs = col_NANs[len(col_NANs)-len_NANs:]
    else:
      col_NANs = col_NANs[len(col_NANs)-top:]
    col_NANs = col_NANs/self.n_all
    if bar:
      col_NANs.plot.bar(figsize=(7,3))
      plt.show()
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[col_NANs.index]
      self.viewF('custom',significance=0.01)
      self.x_all = tmp.copy()   
    if get_return:
      return col_NANs[::-1]
 
 
    
  def Skews(self, top=10, bar=True, plot=False,get_return=False):
    col_Skews = self.x_all.skew().sort_values()
    if top==-1:
      pass
    else:
      col_Skews = col_Skews.iloc[len(col_Skews)-top:]
    if bar:
      col_Skews.plot.bar(figsize=(7,3))
      plt.show()
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[col_Skews.index]
      self.viewF('custom',significance=0.01)
      self.x_all = tmp.copy()   
    if get_return:
      return col_Skews[::-1]
    
  
    
  def Kurts(self, top=10, bar=True, plot=False, get_return=False):
    col_Kurts = self.x_all.kurt().sort_values()
    if top==-1:
      pass
    else:
      col_Kurts = col_Kurts.iloc[len(col_Kurts)-top:]
    if bar:
      col_Kurts.plot.bar(figsize=(7,3))
      plt.show()
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[col_Kurts.index]
      self.viewF('custom',significance=0.01)
      self.x_all = tmp.copy()   
    if get_return:
      return col_Kurts[::-1]
    
    
    
  def Vars(self, bottom=10, bar=True, plot=False, get_return=False):
    col_Vars = self.x_all.var().sort_values()
    if bottom==-1:
      pass
    else:
      col_Vars = col_Vars.iloc[:bottom]
    if bar:
      col_Vars.plot.bar(figsize=(7,3))
      plt.show()
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[col_Vars.index]
      self.viewF('custom',significance=0.01)
      self.x_all = tmp.copy()   
    if get_return:
      return col_Vars[::-1]
  
    
    
  def CorrY(self, view_set=True,view_set_top=10, bar=True, plot=False, get_return=False, input_cols=[]):
    columns = self.col_numeric
    if len(input_cols)!=0:
      columns = input_cols
    corrs = []
    for index in columns:
      df_tmp = pd.DataFrame({index:self.x_train[index],'|corr|':self.y_train})
      corr = df_tmp.corr()[index][1] 
      corrs.append(abs(corr))
    df_corr = pd.DataFrame({'Features':columns,'|corr|':corrs})
    df_corr = df_corr.set_index('Features')
    df_corr = df_corr.sort_values('|corr|',ascending=False)
    
    if view_set:
      print(df_corr.head(view_set_top))
    
    if bar:
      df_corr.plot.bar(figsize=(17,3))
      plt.show()
      
    if plot:
      tmp = self.x_all.copy()
      self.x_all = self.x_all[df_corr.index]
      self.viewF('custom',significance=0.01)
      self.x_all = tmp.copy()   
      
    if get_return:
      return df_corr
  
     
  
  def CorrF(self, method='pearson', view_set=True, data='all', view_set_top=10, view_map=False, get_return=False):
    data_set = (
              'all', 
              'train',
              'test'
              )
    if data is 'all':
      df_corr = self.x_all.corr(method)
    if data is 'train':
      df_corr = self.x_train.corr(method)
    if data is 'test':
      df_corr = self.x_test.corr(method)
    else:
      print('''
      Error! 'data' must be in {}
      '''.format(data_set))
      
    df_corr = abs(df_corr)
    n = len(df_corr)
    corr_ary = []
    var1_ary = []
    var2_ary = []
    for i in range(n):
      for j in range(i):
        if i==j:
          continue
        corr_ary.append(df_corr.iloc[i,j])
        var1_ary.append(df_corr.columns[i])
        var2_ary.append(df_corr.columns[j])
    df_new = pd.DataFrame([])
    df_new["var1"] = var1_ary
    df_new["var2"] = var2_ary
    df_new["|corr|"] = corr_ary
    df_new = df_new.sort_values('|corr|',ascending=False).reset_index(drop=True)
    
    if view_set:
      print(df_new.head(view_set_top))
    
    if view_map:
      sns.heatmap(df_corr, vmax=1, vmin=0, center=0)
      plt.show()
      
    if get_return:
      return df_new
    
    

  def unique(self, input_cols=[]):
    if len(input_cols)==0:
      for col in self.col_object:
        print(self.x_all[col].value_counts(),'\n')
    else:
      for col in input_cols:
        print(self.x_all[col].value_counts(),'\n')