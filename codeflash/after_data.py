def check_missing_data(data: pd.DataFrame):
    """Check if there is any missing data in the DataFrame"""
    return data.isnull().values.any()
