def check_missing_data(data: pd.DataFrame):
    """Check if there is any missing data in the DataFrame"""
    missing_data = data.isnull().sum().sum() > 0
    return missing_data
