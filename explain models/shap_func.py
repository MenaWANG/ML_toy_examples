import shap
import numpy as np
import pandas as pd

def get_SHAP_importance(model,X):
    """This function computes SHAP (SHapley Additive exPlanations) importance values
    for a given machine learning model and input data. It can be used for both
    classification and regression models.
    
    Args:
    - model (object): The trained machine learning model.
    - X (DataFrame): The input data used for SHAP value calculation.
    
    Returns:
    DataFrame: A DataFrame containing three columns:
    - 'feature_name': The names of the input features.
    - 'mean_absolute_shap_value': The mean absolute SHAP value for each feature.
    - 'max_shap_magnitude': The maximum SHAP magnitude for each feature.
    
    Example:
    >>> model = RandomForestRegressor()
    >>> model.fit(X_train, y_train)
    >>> shap_df = get_SHAP_importance(model, X_new)
    >>> print(shap_df.head())
    """
    
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)
    
    if  isinstance(shap_values, list):
        shap_values=shap_values[1]
    
    mag_shap = np.abs(shap_values)
    mean_shaps = np.mean(mag_shap,axis=0)
    
    mean_shap_df = pd.DataFrame(mean_shaps,columns = ['mean_absolute_shap_value'])
    mean_shap_df['max_shap_magnitude'] = np.max(mag_shap,axis=0)
    mean_shap_df['feature_name'] = X.columns
    mean_shap_df.sort_values(by='mean_absolute_shap_value', ascending=False, inplace = True)
    
    return mean_shap_df[['feature_name','mean_absolute_shap_value','max_shap_magnitude']]