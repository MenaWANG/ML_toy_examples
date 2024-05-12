import pandas as pd
import numpy as np
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.feature_selection import RFE, RFECV
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow


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
    def get_best_fold(fold_log, metric, best_set_name):
        """This function find the fold that performed the best on a selected metric 
        and return the target best_set for that fold
        """
        best_fold = None
        best_score = 0.0  
        metric_values = []  # store metric values for all folds

        for fold, data in fold_log.items():
            score = data[metric]
            metric_values.append(score)

            if score > best_score:
                best_score = score
                best_fold = fold

        std_metric = np.std(metric_values)
        print(f'The best score is {best_score:.3}, and the std is {std_metric:.3}')

        return fold_log[best_fold][best_set_name]
    
    @staticmethod
    def gini_scorer(y_true, y_prob):
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        return gini
    
    @staticmethod
    def fbeta_scorer(y_true, y_prob, beta=1):
        fbeta = fbeta_score(y_true, y_prob, beta)
        return fbeta    
    
    @staticmethod
    def plot_cat_variables(df, categorical_columns,num_plots_per_row=3):
        """
        Generate a combined plot with multiple subplots for categorical variables in a DataFrame using Seaborn.

        Parameters:
        - df: DataFrame
        - categorical_columns: list of str, names of categorical variable columns in the DataFrame
        - num_plots_per_row: number of plots per row
        """
        num_columns = len(categorical_columns)
        num_rows = (num_columns + 2) // num_plots_per_row

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_plots_per_row,figsize=(15, num_rows * 4))

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        for i, variable in enumerate(categorical_columns):
            # Create a bar plot using Seaborn
            sns.countplot(x=variable, data=df, color = "#0099DD", ax=axes[i])
            axes[i].set_xlabel(f'{variable.capitalize()}')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'Univariate Bar Plot for {variable.capitalize()}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)

        # Hide empty subplots if any
        for j in range(i + 1, num_rows * num_plots_per_row):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_num_variables(df, numeric_columns, num_plots_per_row=2):
        """
        Generate a combined plot with multiple subplots for numeric variables in a DataFrame using Seaborn.

        Parameters:
        - df: DataFrame
        - numeric_columns: list of str, names of numeric variable columns in the DataFrame
        - num_plots_per_row: number of plots per row
        """
        num_columns = len(numeric_columns)
        num_rows = (num_columns + 2) // num_plots_per_row

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, num_rows * 4))

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        for i, variable in enumerate(numeric_columns):
            # Create a histogram using Seaborn
            sns.histplot(data=df, x=variable, color = "#0099DD", ax=axes[i], kde=True)
            axes[i].set_xlabel(f'{variable.capitalize()}')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Histogram for {variable.capitalize()}', fontweight='bold')

        # Hide empty subplots if any
        for j in range(i + 1, num_rows * num_plots_per_row):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    @staticmethod    
    def scatterplot_highlight(X, highlight, num_plots_per_row=3):
        """
        This function takes X and identified outliers or anomalies using any desired algorithm and 
        creates scatter plots to highlight these outliers for all feature combinations in X.

        Parameters:
        - X: DataFrame to be analyzed, it should only contains numeric features
        - highlight: Array with two values, 0 and 1, with 1 representing cases to be highlighted
        - num_plots_per_row: number of scatter plots to display per row (default is 3)
        """
        
        num_features = X.shape[1]

        # Generate all combinations of feature indices
        feature_combinations = list(itertools.combinations(range(num_features), 2))

        # Calculate the number of rows needed for the subplots
        num_rows = len(feature_combinations) // num_plots_per_row
        if len(feature_combinations) % num_plots_per_row != 0:
            num_rows += 1

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, 5 * num_rows))

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        for i, (feature1, feature2) in enumerate(feature_combinations):
            # Calculate subplot position
            row = i // num_plots_per_row
            col = i % num_plots_per_row

            # Scatter plot for the current feature combination
            sns.scatterplot(x=X.iloc[:, feature1], y=X.iloc[:, feature2], 
                            hue=highlight, palette={0: 'green', 1: 'red'}, 
                            ax=axes[i])
            axes[i].set_title(f'Scatterplot: {X.columns[feature2]} vs {X.columns[feature1]}')

        # Adjust layout and show plots
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def is_primary_key(df, cols):
        """
        Check if the combination of specified columns forms a primary key in the DataFrame.

        Args:
            df (DataFrame): The DataFrame to check.
            cols (list): A list of column names to check for forming a primary key.

        Returns:
            bool: True if the combination of columns forms a primary key, False otherwise.
        """
        # Check if the DataFrame is not empty
        if df.empty:
            print("DataFrame is empty.")
            return False

        # Check if all columns exist in the DataFrame
        missing_cols = [col_name for col_name in cols if col_name not in df.columns]
        if missing_cols:
            print(f"Columns {', '.join(missing_cols)} do not exist in the DataFrame.")
            return False

        # Check for missing values in each specified column
        for col_name in cols:
            missing_rows_count = df[col_name].isnull().sum()
            if missing_rows_count > 0:
                print(f"There are {missing_rows_count:,} row(s) with missing values in column '{col_name}'.")

        # Filter out rows with missing values in any of the specified columns
        filtered_df = df.dropna(subset=cols)

        # Check if the combination of columns is unique after filtering out missing value rows
        unique_row_count = filtered_df.duplicated(subset=cols).sum()
        total_row_count = len(filtered_df)

        print(f"Total row count after filtering out missings: {total_row_count:,}")
        print(f"Unique row count after filtering out missings: {total_row_count - unique_row_count:,}")

        if unique_row_count == 0:
            print(f"The column(s) {', '.join(cols)} forms a primary key.")
            return True
        else:
            print(f"The column(s) {', '.join(cols)} does not form a primary key.")
            return False
    
    @staticmethod
    def find_duplicates(df, cols):
        """
        Function to find duplicate rows in a Pandas DataFrame based on specified columns.

        Args:
        - df: Pandas DataFrame
        - cols: List of column names to check for duplicates

        Returns:
        - duplicates: Pandas DataFrame containing duplicate rows based on the specified columns,
                    with the specified columns and the 'count' column as the first columns,
                    along with the rest of the columns from the original DataFrame
        """
        # Filter out rows with missing values in any of the specified columns
        df = df.dropna(subset=cols)

        # Group by the specified columns and count the occurrences
        dup_counts = df.groupby(cols).size().reset_index(name='count')
        
        # Filter to retain only the rows with count greater than 1
        duplicates = dup_counts[dup_counts['count'] > 1]
        
        # Join with the original DataFrame to include all columns
        duplicates = pd.merge(duplicates, df, on=cols, how='inner')
        
        # Reorder columns with 'count' as the first column
        duplicate_cols = ['count'] + cols
        duplicates = duplicates[duplicate_cols + [c for c in df.columns if c not in cols]]
        
        return duplicates



    @staticmethod
    def cols_responsible_for_id_dups(df, id_list):
        """
        Warning: This diagnostic function may take a long time to run.
        
        This function checks each non-ID column for each unique combination of ID columns to detect differences. 
        It generates a summary table indicating the columns and their respective difference counts 
        when the specified ID columns have the same values. This can help identify columns 
        responsible for duplicates for any given ID combinations in the id_list.
        
        Please note that this process involves checking each non-ID column against all unique combinations 
        of ID columns, which can be time-consuming for large datasets.
        
        Args:
        - df (DataFrame): The Pandas DataFrame to analyze.
        - id_list (list): A list of column names representing the ID columns.

        Returns:
        - summary_table (DataFrame): A Pandas DataFrame containing two columns - 'col_name' and 
        'difference_counts'. 'col_name' represents the column name, and 'difference_counts' 
        represents the count of differing values for each column when ID columns have the same values.
        """
        # Filter out rows with missing values in any of the ID columns
        filtered_df = df.dropna(subset=id_list)
        
        # Function to count differences between columns
        def count_differences(col_name):
            """
            Counts the number of differing values for a given column when ID columns have the same values.

            Args:
            - col_name (str): The name of the column to analyze.

            Returns:
            - count (int): The count of differing values.
            """
            # Group by the ID columns and the current column, then count distinct values
            distinct_count = filtered_df.groupby(id_list + [col_name]).size().reset_index().groupby(id_list).size().reset_index(name="count")
            return distinct_count[distinct_count["count"] > 1]["count"].count()
        
        # Get the column names excluding the ID columns
        value_cols = [col_name for col_name in df.columns if col_name not in id_list]
        
        # Create a DataFrame to store the summary table
        summary_data = [(col_name, count_differences(col_name)) for col_name in value_cols]
        summary_table = pd.DataFrame(summary_data, columns=["col_name", "difference_counts"])
        
        # Sort the summary_table by "difference_counts" from large to small
        summary_table = summary_table.sort_values(by="difference_counts", ascending=False)
        
        return summary_table




class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bin_features=[], num_features=[], 
                 cat_features=[], binning_settings=None):
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

class FeatureSelector:
    """This feature selector takes any estimator, scoring and CV methods 
    and feature selection configurations and return the best set selectioned through RFE. 
    """
    def __init__(self, estimator, min_features_to_select=2, step=1, scorer=None, cv=None):
        """
        Initialize the FeatureSelector.

        Args:
            estimator: The estimator to be used for feature selection, e.g., a classifier or regressor.
            n_features_to_select: The number of features to select. If None, half of the features will be selected.
            step: The number of features to remove at each iteration (default is 1).
            cv: Cross-validation generator (default is StratifiedKFold with 5 folds).
        """
        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.scorer = scorer
        self.cv = cv

    def select_features(self, X, y):
        """
        Select the best features using RFE (Recursive Feature Elimination) and cross-validation.

        Parameters:
            X: The feature matrix.
            y: The target values.

        Returns:
            selected_features: The selected feature indices.
        """
        self.rfecv = RFECV(estimator=self.estimator, 
                  min_features_to_select=self.min_features_to_select, 
                  step=self.step,
                  scoring=self.scorer,
                  cv=self.cv)
        selected_feature_indices = self.rfecv.fit(X, y).support_
        selected_features = X.loc[:, selected_feature_indices].columns.tolist()
        cv_score = np.max(self.rfecv.cv_results_['mean_test_score'])

        return selected_features, cv_score
    
    def summary_plot(self):
        n_scores = len(self.rfecv.cv_results_["mean_test_score"])
        feature_numbers = range(self.min_features_to_select, n_scores + self.min_features_to_select)

        # Find the index of the maximum mean test score
        max_score_index = np.argmax(self.rfecv.cv_results_["mean_test_score"])
        max_score = self.rfecv.cv_results_["mean_test_score"][max_score_index]
        print(f'the max test score is {max_score:.3} achieved with {max_score_index+self.min_features_to_select} features')
        
        # Calculate the y-axis limits
        min_y = min(self.rfecv.cv_results_["mean_test_score"] - self.rfecv.cv_results_["std_test_score"])
        max_y = max(self.rfecv.cv_results_["mean_test_score"] + self.rfecv.cv_results_["std_test_score"])
        
        # Plot the mean test scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test score")
        plt.errorbar(
            feature_numbers,
            self.rfecv.cv_results_["mean_test_score"],
            yerr=self.rfecv.cv_results_["std_test_score"],
            label="Mean Test Score",
        )

        # Highlight the bar with the highest score by covering the entire y-axis range
        plt.bar(
            feature_numbers[max_score_index],
            max_y - min_y,  # Bar height
            bottom=min_y,   # Set the bottom of the bar to the minimum y-axis value
            color='red',    
            alpha=0.3,      
            label="Max Test Score",
        )

        # Add a textbox at the bottom right with information about the maximum score and number of features
        text_str = f"Max Score: {max_score:.3f}\n# of Features: {feature_numbers[max_score_index]}"
        plt.text(0.98, 0.02, text_str, transform=plt.gca().transAxes,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Set y-axis limits for better visibility
        plt.ylim(min_y, max_y)

        plt.title("Feature Selection\nScore by number of features")
        # plt.legend()
        plt.show()

class ModelPipeline(mlflow.pyfunc.PythonModel):
    """This class takes a model with the optimized configuration 
        (e.g., algorithm, hyperparameters, feature_set), fit it on the train data,
        and then be ready to serve and explain its predictions.

    Args:
        custom_transformer: A preprocessing pipeline that can be trained to preprocess the data
        model: any algorithm that follows sklearn API which hyperparameters has been optimized for the problem
        # feature_set, let's add this at the next iteration
        # TODO: maybe in future we will prefer to pass custom_transformer_params and model_params into the mlops_class pipeline, so it is easier for our user to use
    """

    def __init__(self, custom_transformer=None, model=None):
        self.custom_transformer = custom_transformer
        self.model = model
    
    def fit(self, X, y=None):
        """Fit the preprocessing transformer and the model using training data.

    Args:
        X (DataFrame): Training data to fit the preprocessing transformer and the model.
        y (Series): Target labels. 
        """
        X_train_transformed = self.custom_transformer.fit_transform(X, y)
        self.model.fit(X_train_transformed, y)
    
    def predict_proba(self,X):
        X_new_transformed = self.custom_transformer.transform(X)
        return self.model.predict_proba(X_new_transformed)
        
    def explain_model(self,X):
        X_transformed = self.custom_transformer.transform(X.copy())
        self.X_explain = X_transformed.copy()
        # get the original values for binned and numeric features to show when explaining cases
        self.X_explain[self.custom_transformer.bin_features + self.custom_transformer.num_features] = X[self.custom_transformer.bin_features + self.custom_transformer.num_features]
        self.X_explain.reset_index(drop=True)
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer(X_transformed)  
        try:
            self.shap_values.values.shape[2] 
            self.both_class = True
        except:
            self.both_class = False
        if self.both_class:
            shap.summary_plot(self.shap_values[:,:,1])
        elif self.both_class == False:
            shap.summary_plot(self.shap_values)
    
    def explain_case(self,n):
        if self.shap_values is None:
            print("pls explain model first")
        else:
            self.shap_values.data = self.X_explain
            if self.both_class:
                shap.plots.waterfall(self.shap_values[:,:,1][n-1])
            elif self.both_class == False:
                shap.plots.waterfall(self.shap_values[n-1])