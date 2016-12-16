# Field Name        Definition
# RefID               Unique (sequential) number assigned to vehicles
# IsBadBuy        Identifies if the kicked vehicle was an avoidable purchase 
# PurchDate       The Date the vehicle was Purchased at Auction
# Auction         Auction provider at which the  vehicle was purchased
# VehYear         The manufacturer's year of the vehicle
# VehicleAge        The Years elapsed since the manufacturer's year
# Make          Vehicle Manufacturer 
# Model         Vehicle Model
# Trim          Vehicle Trim Level
# SubModel        Vehicle Submodel
# Color         Vehicle Color
# Transmission        Vehicles transmission type (Automatic, Manual)
# WheelTypeID       The type id of the vehicle wheel
# WheelType       The vehicle wheel type description (Alloy, Covers)
# VehOdo          The vehicles odometer reading
# Nationality       The Manufacturer's country
# Size          The size category of the vehicle (Compact, SUV, etc.)
# TopThreeAmericanName      Identifies if the manufacturer is one of the top three American manufacturers
# MMRAcquisitionAuctionAveragePrice Acquisition price for this vehicle in average condition at time of purchase 
# MMRAcquisitionAuctionCleanPrice   Acquisition price for this vehicle in the above Average condition at time of purchase
# MMRAcquisitionRetailAveragePrice  Acquisition price for this vehicle in the retail market in average condition at time of purchase
# MMRAcquisitonRetailCleanPrice   Acquisition price for this vehicle in the retail market in above average condition at time of purchase
# MMRCurrentAuctionAveragePrice   Acquisition price for this vehicle in average condition as of current day 
# MMRCurrentAuctionCleanPrice   Acquisition price for this vehicle in the above condition as of current day
# MMRCurrentRetailAveragePrice    Acquisition price for this vehicle in the retail market in average condition as of current day
# MMRCurrentRetailCleanPrice    Acquisition price for this vehicle in the retail market in above average condition as of current day
# PRIMEUNIT       Identifies if the vehicle would have a higher demand than a standard purchase
# AcquisitionType       Identifies how the vehicle was aquired (Auction buy, trade in, etc)
# AUCGUART        The level guarntee provided by auction for the vehicle (Green light - Guaranteed/arbitratable, Yellow Light - caution/issue, red light - sold as is)
# KickDate        Date the vehicle was kicked back to the auction
# BYRNO         Unique number assigned to the buyer that purchased the vehicle
# VNZIP                                   Zipcode where the car was purchased
# VNST                                    State where the the car was purchased
# VehBCost        Acquisition cost paid for the vehicle at time of purchase
# IsOnlineSale        Identifies if the vehicle was originally purchased online
# WarrantyCost                            Warranty price (term=36month  and millage=36K) 


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

class LemonCarFeaturizer():
  def __init__(self):
    vectorizer = None
    self._imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    self._binarizer = preprocessing.Binarizer()
    self._scaler = preprocessing.MinMaxScaler()
    self._preprocs = [self._imputer, \
                      #self._binarizer, \
                      #self._scaler
                      ]

  def _fit_transform(self, dataset):
    for p in self._preprocs:
      dataset = self._proc_fit_transform(p, dataset)
    return dataset

  def _transform(self, dataset):
    for p in self._preprocs:
      dataset = p.transform(dataset)

    return dataset

  def _proc_fit_transform(self, p, dataset):
    p.fit(dataset)
    dataset = p.transform(dataset)
    return dataset

  def create_features(self, dataset, training=False):
    grouped_keys = ['TranSizePair','Model','SubModel','VNST','PRIMEUNIT','AUCGUART','TopThreeAmericanName','Size','Nationality','WheelType','Transmission','Color','Trim','Make','Auction','PurchDate']
    for gkey in grouped_keys:
      if gkey in dataset:
        try:
          dataset[gkey] = pd.Categorical.from_array(dataset[gkey]).codes
        except:
          import IPython; IPython.embed()
    # data = dataset[ [
    #               'MMRAcquisitonRetailCleanPrice',
    #               'MMRCurrentAuctionAveragePrice',
    #               'MMRCurrentAuctionCleanPrice',
    #               'MMRCurrentRetailAveragePrice',
    #               'VehYear',
    #               'VehOdo',
    #               #'KickDate',
    #               'MMRCurrentRetailCleanPrice']
    #       ]
    # #data ['']
    # data = dataset[ ['PurchDate', 'VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Transmission', 'WheelTypeID', 'WheelType', 'VehOdo', 'Size', 'TopThreeAmericanName', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VehBCost'] ]
    # data = dataset[['PurchDate', 'PurchDate', 'VehYear', 'Trim', 'AUCGUART', 'PurchDate', 'VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Transmission', 'WheelTypeID', 'WheelType', 'VehOdo', 'Nationality', 'Size', 'TopThreeAmericanName', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VehBCost']]
    data = dataset[[
    'AUCGUART',
    'AnnMileage',
    'AucClnPrAgeRatio',
    'BYRNO',
    'Color',
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRCurrentRetailAveragePrice',
    'Make',
    'Model',
    'Nationality',
    'PDiffACP',
    'PDiffRAP',
    'PDiffRCP',
    'PRIMEUNIT',
    'RetAvgPrAgeRatio',
    'RetClnPrAgeRatio',
    'SubModel',
    'TopThreeAmericanName',
    'TranSizePair',
    'Trim',
    'VehBCost',
    'VehOdo',
    'VehicleAge',
    'WheelTypeID',
    'MMRCurrentAuctionCleanPrice',
    'Size',
    'VehYear']]
    if training:
      data = self._fit_transform(data)
    else:
      data = self._transform(data)
    return data

def train_model(X, y,cfier):
  if cfier == 0:
    print("\tRidge")
    model = RidgeClassifierCV()
  elif cfier == 1:
    print("\tLogReg")
    model = LogisticRegression(C=10)
  elif cfier == 2:
    print("\tDecTree")
    model = DecisionTreeClassifier()
  elif cfier == 3: 
    print("\tRandForest")
    model = RandomForestClassifier()
  elif cfier == 4:
    print("\tGradBoostC")
    model = GradientBoostingClassifier(n_estimators=175, learning_rate=0.085, max_depth=3, random_state=0)
  elif cfier == 5:
    print('\tAda')
    model = AdaBoostClassifier()
  elif cfier == 6:
    print("\tGradientBoostingRegressor")
    model = GradientBoostingRegressor()
  elif cfier == 7:
    print("\tExtraTreesClassifier")
    model = ExtraTreesClassifier()
  elif cfier == 8:
    print("\tExtraTreesRegressor")
    model = ExtraTreesRegressor()
  elif cfier == 9:
    print("\tKNeighborsClassifier - 5")
    model = KNeighborsClassifier(n_neighbors=3)
  elif cfier == 10:
    print("\tSVC")
    model = SVC()
  elif cfier == 11:
    logr = LogisticRegression(C=10)
    # rdge = RidgeClassifierCV()
    rfor = RandomForestClassifier()
    grad = GradientBoostingClassifier()
    adac = AdaBoostClassifier()
    etcl = ExtraTreesClassifier()
    dect = DecisionTreeClassifier()
    # grdr = GradientBoostingRegressor()
    # etrg = ExtraTreesRegressor()
    # kngh = KNeighborsClassifier(n_neighbors=5)
    model = VotingClassifier(estimators=[('lr',logr),('rfc',rfor),('gbc',grad),('abc',adac),('etc',etcl),('dtc',dect)],voting='soft',weights=[5.02,5.44,6,5.96,5.36,4.68])
  model.fit(X, y)
  #print model.coef_
  return model

def predict(model, y):
  return model.predict(y)

def create_submission(model, transformer):
  submission_test = pd.read_csv('inclass_test.csv')
  submission_test = add_fields(submission_test)
  grouped_keys = ['TranSizePair','Model','SubModel','VNST','PRIMEUNIT','AUCGUART','TopThreeAmericanName','Size','Nationality','WheelType','Transmission','Color','Trim','Make','Auction','PurchDate']
  for gkey in grouped_keys:
    if gkey in submission_test:
      try:
        submission_test[gkey] = pd.Categorical.from_array(submission_test[gkey]).codes
      except:
        import IPython; IPython.embed()
  predictions = pd.Series([x[1] for x in model.predict_proba(transformer.create_features(submission_test))])

  submission = pd.DataFrame({'RefId': submission_test.RefId, 'IsBadBuy': predictions})
  submission.sort_index(axis=1, inplace=True)
  submission.to_csv('submission.csv', index=False)

def add_fields(data):
  data['AnnMileage'] = data['VehOdo']/data['VehicleAge']
  data['AucAvgPrAgeRatio'] = data['MMRAcquisitionAuctionAveragePrice']/data['VehicleAge']
  data['AucClnPrAgeRatio'] = data['MMRAcquisitionAuctionCleanPrice']/data['VehicleAge']
  data['RetAvgPrAgeRatio'] = data['MMRAcquisitionRetailAveragePrice']/data['VehicleAge']
  data['RetClnPrAgeRatio'] = data['MMRAcquisitonRetailCleanPrice']/data['VehicleAge']
  data['TranSizePair'] = data['Transmission']+" - "+data['Size']
  data['PDiffAAP'] = data['VehBCost'] - data['MMRAcquisitionAuctionAveragePrice']
  data['PDiffACP'] = data['VehBCost'] - data['MMRAcquisitionAuctionCleanPrice']
  data['PDiffRAP'] = data['VehBCost'] - data['MMRAcquisitionRetailAveragePrice']
  data['PDiffRCP'] = data['VehBCost'] - data['MMRAcquisitonRetailCleanPrice']
  return data

def main():
  data = pd.read_csv('inclass_training.csv')
  data = add_fields(data)
  #print data
  featurizer = LemonCarFeaturizer()
  
  print ("Transforming dataset into features...")
  X = featurizer.create_features(data, training=True)
  y = data.IsBadBuy

  for i in range(4,5):
    print ("\tTraining model...")
    model = train_model(X,y,i)

    print ("\tCross validating...")
    print ("\t"+str(np.mean(cross_val_score(model, X, y, scoring='roc_auc'))))

  print ("Create predictions on submission set...")
  create_submission(model, featurizer)


if __name__ == '__main__':
  main()
