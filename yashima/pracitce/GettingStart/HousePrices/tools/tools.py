import pandas as pd

class Core():

  def __init__(self, x_train, x_test, y_train):
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.x_all = pd.concat([self.x_train, self.x_test], sort=False, ignore_index=True, axis=0)

    self.n_train = len(self.x_train)
    self.n_test = len(self.x_test)

    self.col_object = self.x_train.select_dtypes(include='object')
    self.col_int = self.x_train.select_dtypes(include='int')
    self.col_float = self.x_train.select_dtypes(include='float')

    self.missing = self.x_all.isnull().sum()

  def loadDF(self, x_train, x_test, y_train):
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.x_all = pd.concat([self.x_train, self.x_test], sort=False, ignore_index=True, axis=0)

    self.n_train = len(self.x_train)
    self.n_test = len(self.x_test)

    self.col_object = self.x_train.select_dtypes(include='object')
    self.col_int = self.x_train.select_dtypes(include='int')
    self.col_float = self.x_train.select_dtypes(include='float')

    self.missing = self.x_all.isnull().sum()

  def split(self):
    self.x_train = self.x_all.iloc[self.n_train:]
    self.x_test = self.x_all.iloc[:self.n_train]

class Preprocess(Core):
  
  def __init__(self, x_train, x_test, y_train):
    super().__init__(x_train, x_test, y_train)
  
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