from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer 


# Function to impute binary and continuous variables together
def impute_combined_data(data, binary_cols, continuous_cols):

    # Reset the index to prevent errors
    data = data.reset_index(drop=True)

    # Create a column-specific logic for binary columns
    def round_binary_columns(data_array, binary_col_indices):
        for idx in binary_col_indices:
            data_array[:, idx] = (data_array[:, idx] > 0.5).astype(int)
        return data_array

    # Combine all columns for imputation
    all_cols = binary_cols + continuous_cols
    binary_indices = [all_cols.index(col) for col in binary_cols]

    # Create a unified imputer using RandomForestRegressor
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), max_iter=10)

    # Fit and transform the entire dataset
    imputed_array = imputer.fit_transform(data[all_cols])

    # Ensure binary columns are rounded to 0 or 1
    imputed_array = round_binary_columns(imputed_array, binary_indices)

    # Convert back to DataFrame
    imputed_data = pd.DataFrame(imputed_array, columns=all_cols)
    return imputed_data
