import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from optbinning import OptimalBinning

class MLUtils:
    @staticmethod
    def group_train_test_split(X, y, group, test_size=0.2, random_state=42):
        """
        Split a dataset into training and testing sets while ensuring that members of the same group remain together in either the training or testing set, i.e., they are not separated.

        Args:
            X (DataFrame): Features or input data.
            y (array-like): Target values or labels.
            group (array-like): Array representing group memberships for each data point.
            test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
            random_state (int): Seed for the random number generator (default is 42).

        Returns:
            X_train (DataFrame) 
            X_test (DataFrame)
            y_train (array-like)
            y_test (array-like)
        """
        gss = GroupShuffleSplit(n_splits=1, test_size = test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, group))
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bin_features=None, num_features=None, 
                 cat_features=None, binning_settings=None):
        self.bin_features = bin_features
        self.num_features = num_features
        self.cat_features = cat_features
        self.binning_settings = binning_settings
        self.binning_models = {}
        self.binning_tables = {}
        self.num_transformer = None
        self.cat_transformer = None
        self.transformed_cat_cols = []

    def fit(self, X, y=None):
        if self.bin_features:
            for feature_name in self.bin_features:
                bin_cfg = self.binning_settings[feature_name]
                print(f'configuration for {feature_name}:', bin_cfg)
                binning_model = OptimalBinning(name=feature_name,
                                               special_codes = bin_cfg.get('special_codes', None),
                                               user_splits = bin_cfg.get('user_splits', None),
                                               dtype = bin_cfg.get('dtype', 'numerical'),
                                               monotonic_trend = bin_cfg.get('monotonic_trend', None))
                binning_model.fit(X[feature_name], y)
                self.binning_models[feature_name] = binning_model
                self.binning_tables[feature_name] = binning_model.binning_table

        if self.num_features:
            self.num_transformer = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy = 'median'))
            ])
            self.num_transformer.fit(X[self.num_features])
        
        if self.cat_features:
            self.cat_transformer = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            self.cat_transformer.fit(X[self.cat_features])
        
        return self
    
    def get_transformed_cat_cols(self):
        cat_cols = []
        cats = self.cat_features
        cat_values = self.cat_transformer['encoder'].categories_
        for cat, values in zip(cats, cat_values):
            cat_cols += [f'{cat}_{value}' for value in values]
        
        return cat_cols
        
    
    def transform(self, X):
        X_transformed = pd.DataFrame()

        if self.bin_features:
            for feature_name in self.bin_features:
                binning_model = self.binning_models.get(feature_name, None)
                X_transformed[feature_name] = binning_model.transform(X[feature_name])

        if self.num_features:
            transformed_num_data = self.num_transformer.transform(X[self.num_features])
            X_transformed[self.num_features] = transformed_num_data
        
        if self.cat_features:
            transformed_cat_data = self.cat_transformer.transform(X[self.cat_features]).toarray()
            self.transformed_cat_cols = self.get_transformed_cat_cols()
            transformed_cat_df = pd.DataFrame(transformed_cat_data,
                                            columns = self.transformed_cat_cols)      
            X_transformed = pd.concat([X_transformed,transformed_cat_df], axis=1)
        
        X_transformed.index = X.index

        return X_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)