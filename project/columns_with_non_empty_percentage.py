def columns_with_non_empty_percentage(df, percentage):
    """
    Returns the columns that contain more than a specified percentage of non-empty fields.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    percentage (float): The percentage threshold (between 0 and 100).

    Returns:
    list: List of column names that have more than the specified percentage of non-empty fields.
    """
    # Calculate the threshold as a fraction
    threshold = percentage

    # Calculate the percentage of non-empty fields for each column
    non_empty_percentages = df.notna().mean()

    # Filter columns based on the threshold
    selected_columns = non_empty_percentages[non_empty_percentages >= threshold].index.tolist()

    return selected_columns