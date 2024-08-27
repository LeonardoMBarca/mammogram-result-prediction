from sklearn.preprocessing import StandardScaler


def preprocess_data(dataframe):
    """
    Main function to preprocess the DataFrame.

    Args:
    dataframe: DataFrame containing the data to be processed.

    Returns:
    dict: A dictionary containing the transformed features and the target.
    """
    transformed_data = data_transformation(dataframe)
    features = transformed_data["features"]
    target = transformed_data["target"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return {"features": features_scaled, "target": target}

def data_transformation(dataframe):
    """
    Function to transform DataFrame data into numpy arrays.

    Args:
    dataframe: DataFrame containing the data to be transformed.

    Returns:
    dict: A dictionary containing the features and the target.
    """
    features_transform = dataframe[["Age", "Shape", "Margin", "Density"]].values
    target_transform = dataframe["Severity"].values

    return {"features": features_transform, "target": target_transform}