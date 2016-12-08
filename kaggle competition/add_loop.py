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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# import argparse
from itertools import combinations 
from random import shuffle

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

  def create_features(self, data, keys, training=False):
    # import IPython; IPython.embed()
    # data = dataset[ [
    #               # 'MMRAcquisitonRetailCleanPrice',
    #               'MMRAcquisitionRetailAveragePrice',
    #               'MMRCurrentAuctionAveragePrice',
    #               # 'MMRCurrentAuctionCleanPrice',
    #               'MMRAcquisitionAuctionAveragePrice',
    #               # 'MMRAcquisitionAuctionCleanPrice',
    #               'MMRCurrentRetailAveragePrice',
    #               # 'MMRCurrentRetailCleanPrice',
    #               'VehYear',
    #               'VehOdo',
    #               'Make',
    #               'TopThreeAmericanName',
    #               #'KickDate',
    #               # 'MMRCurrentRetailCleanPrice',
    #               'WheelTypeID',
    #               'VehBCost',
    #               # 'VehYear',
    #               'VehicleAge',
    #               'Size',
    #               # 'AcquisitionType',
    #               'WarrantyCost',
    #               # 'IsOnlineSale',
    #               # ''
    #               ]
    #       ]
    # data['Make'] = encode(data,'Make')
    # data['TopThreeAmericanName'] = encode(data,'TopThreeAmericanName')
    # data['Size'] = encode(data,'Size')
    # data['AcquisitionType'] = encode(data,'AcquisitionType')
    #data ['']
    data = data[keys]
    if training:
      data = self._fit_transform(data)
    else:
      data = self._transform(data)
    return data


def encode(data,key):
  le = preprocessing.LabelEncoder()
  le.fit(data['Make'])
  return le.transform(data['Make'])

def train_model(X, y,cfier):
  if cfier == 1:
    model = RidgeClassifierCV()
  elif cfier == 2:
    model = LogisticRegression(C=10)
  elif cfier == 3:
    model = DecisionTreeClassifier()
  elif cfier == 4: 
    model = RandomForestClassifier()
  else:
    model = GradientBoostingClassifier()
  model.fit(X, y)
  #print model.coef_
  return model

def predict(model, y):
  return model.predict(y)

def add_fields(data):
  data['AnnMileage'] = data['VehOdo']/data['VehicleAge']
  data['AucAvgPrAgeRatio'] = data['MMRAcquisitionAuctionAveragePrice']/data['VehicleAge']
  data['AucClnPrAgeRatio'] = data['MMRAcquisitionAuctionCleanPrice']/data['VehicleAge']
  data['RetAvgPrAgeRatio'] = data['MMRAcquisitionRetailAveragePrice']/data['VehicleAge']
  data['RetClnPrAgeRatio'] = data['MMRAcquisitonRetailCleanPrice']/data['VehicleAge']
  data['TranSizePair'] = data['Transmission']+" - "+data['Size']
  return data

def create_submission(model, transformer,keys):
  submission_test = pd.read_csv('inclass_test.csv')
  submission_test = add_fields(submission_test)
  grouped_keys = ['TranSizePair','Model','SubModel','VNST','PRIMEUNIT','AUCGUART','TopThreeAmericanName','Size','Nationality','WheelType','Transmission','Color','Trim','Make','Auction','PurchDate']
  # import IPython; IPython.embed(); import sys; sys.exit()
  for gkey in grouped_keys:
    submission_test[gkey] = pd.Categorical.from_array(submission_test[gkey]).codes
  predictions = pd.Series([x[1] for x in model.predict_proba(transformer.create_features(submission_test,keys))])

  submission = pd.DataFrame({'RefId': submission_test.RefId, 'IsBadBuy': predictions})
  submission.sort_index(axis=1, inplace=True)
  submission.to_csv('submission.csv', index=False)

def main():
  data = pd.read_csv('inclass_training.csv')
  data = add_fields(data)
  #print data
  featurizer = LemonCarFeaturizer()
  keys = list(data.keys())
  keys.remove('RefId'); keys.remove('IsBadBuy')
  # import IPython; IPython.embed(); import sys; sys.exit()
  # optkeys = ['PurchDate', 'VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Transmission', 'WheelTypeID', 'WheelType', 'VehOdo', 'Size', 'TopThreeAmericanName', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VehBCost']
  # optkeys = ['PurchDate', 'VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Transmission', 'WheelTypeID', 'WheelType', 'VehOdo', 'Size', 'TopThreeAmericanName', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VehBCost', 'Nationality']
  # optkeys = ['PurchDate', 'VehYear', 'PurchDate', 'VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Transmission', 'WheelTypeID', 'WheelType', 'VehOdo', 'Nationality', 'Size', 'TopThreeAmericanName', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VehBCost']
  # optkeys = ['PurchDate', 'VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Transmission', 'WheelTypeID', 'WheelType', 'VehOdo', 'Size', 'MMRAcquisitionAuctionAveragePrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VehBCost', 'IsOnlineSale', 'AnnMileage']
  # optkeys = ['PurchDate', 'VehicleAge', 'VehBCost', 'PurchDate', 'VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Transmission', 'WheelTypeID', 'WheelType', 'VehOdo', 'Size', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'PRIMEUNIT', 'AUCGUART', 'BYRNO', 'VehBCost', 'IsOnlineSale', 'AnnMileage']

  grouped_keys = ['TranSizePair','Model','SubModel','VNST','PRIMEUNIT','AUCGUART','TopThreeAmericanName','Size','Nationality','WheelType','Transmission','Color','Trim','Make','Auction','PurchDate']
  # import IPython; IPython.embed(); import sys; sys.exit()
  for gkey in grouped_keys:
    data[gkey] = pd.Categorical.from_array(data[gkey]).codes
  max_keys = keys #optkeys
  max_score = 0
  max_cfier = 5
  j = 0
  # X = featurizer.create_features(data,max_keys,training=True)
  # y = data.IsBadBuy
  # model = train_model(X,y,max_cfier)
  # max_score = np.mean(cross_val_score(model, X, y, scoring='roc_auc'))
  # print("\nBase score: " + str(max_score)+"\n")
  improvement_made = True
  while improvement_made:
    shuffle(keys)
    improvement_made = False
    for key in keys:
      print("\t"+str(j)+": "+str(max_score))
      j+=1
      if key in max_keys:
        continue
      max_keys.append(key)
      X = featurizer.create_features(data,max_keys,training=True)
      y = data.IsBadBuy
      new_max = False
      for i in range(5,6):
        model = train_model(X,y,i)
        score = np.mean(cross_val_score(model, X, y, scoring='roc_auc'))
        if score > max_score:
          improvement_made = True
          max_score = score
          max_cfier = i
          new_max = True
      if not new_max:
        max_keys.remove(key)
    print("Best score: " + str(max_score))
    print("Cfier #: "+str(max_cfier))
    print("Features: ")
    print(max_keys)
    X = featurizer.create_features(data,max_keys,training=True)
    y = data.IsBadBuy
    model = train_model(X,y,max_cfier)

    print ("Create predictions on submission set...")
    create_submission(model, featurizer,max_keys)


if __name__ == '__main__':
  main()
