import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path, nrows=None):
    """Load the raw csv dataset."""
    return pd.read_csv(file_path, nrows=nrows)

def clean_data(df, config):
    """
    Clean the dataset:
    - Impute missing values
    - Convert columns to appropriate types and categories
    - Encode categorical variables
    """
    # 0. Keep only the relevant columns to avoid dropping all rows due to NaNs in other columns
    relevant_cols = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
        'verification_status', 'issue_d', 'loan_status', 'purpose', 'title',
        'zip_code', 'addr_state', 'dti', 'earliest_cr_line', 'open_acc',
        'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
        'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'address'
    ]
    existing_relevant_cols = [col for col in relevant_cols if col in df.columns]
    df = df[existing_relevant_cols].copy()
    
    # 1. Map target column (loan_status) to binary
    if 'loan_status' in df.columns:
        df['loan_status'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
    
    # 2. Map home ownership values
    if 'home_ownership' in df.columns:
        df.loc[(df['home_ownership'] == 'ANY') | (df['home_ownership'] == 'NONE'), 'home_ownership'] = 'OTHER'

    # 3. Simplify categorical/numerical records to binary indicators
    def map_pub_rec(number):
        return 0 if number == 0.0 else 1

    def map_mort_acc(number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        return number  # Keeps NaN intact

    def map_pub_rec_bankruptcies(number):
        if number == 0.0:
            return 0
        elif number >= 1.0:
            return 1
        return number

    if 'pub_rec' in df.columns:
        df['pub_rec'] = df['pub_rec'].apply(map_pub_rec)
    if 'mort_acc' in df.columns:
        df['mort_acc'] = df['mort_acc'].apply(map_mort_acc)
    if 'pub_rec_bankruptcies' in df.columns:
        df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].apply(map_pub_rec_bankruptcies)

    # 4. Impute mort_acc using total_acc averages
    if 'mort_acc' in df.columns and 'total_acc' in df.columns:
        total_acc_avg = df.groupby('total_acc')['mort_acc'].mean()
        
        def fill_mort_acc(row):
            val = row['mort_acc']
            if pd.isna(val):
                total_acc = row['total_acc']
                if total_acc in total_acc_avg and not pd.isna(total_acc_avg[total_acc]):
                    return round(total_acc_avg[total_acc])
                return 0.0
            return val
            
        df['mort_acc'] = df.apply(fill_mort_acc, axis=1)

    # 5. Drop any remaining rows with missing values
    df.dropna(inplace=True)

    # 6. Map term to numeric
    if 'term' in df.columns:
        term_values = {' 36 months': 36, ' 60 months': 60}
        df['term'] = df['term'].map(term_values)

    # 7. Extract zip_code from address
    if 'address' in df.columns:
        df['zip_code'] = df['address'].apply(lambda x: x[-5:])
    
    # 8. Convert earliest_cr_line to datetime and extract year
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line']).dt.year

    # 9. Drop irrelevant/duplicate columns defined in config
    cols_to_drop = config['preprocessing']['columns_to_drop']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    # 10. Perform One-Hot Encoding (Dummies)
    dummies_list = config['preprocessing']['dummies']
    existing_dummies = [col for col in dummies_list if col in df.columns]
    
    df = pd.get_dummies(df, columns=existing_dummies, drop_first=True, dtype=float)
    
    if 'zip_code' in df.columns:
        df = pd.get_dummies(df, columns=['zip_code'], drop_first=True, dtype=float)
        
    # Drop any remaining non-numeric (string/object) columns dynamically to prevent scaling errors
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        df.drop(columns=non_numeric_cols, inplace=True)
        
    return df

def filter_outliers(train_df, config):
    """
    Apply outlier filtering on the training dataset based on config thresholds.
    This helps prevent model bias while preserving validation set integrity.
    """
    train_df = train_df.copy()
    outlier_cfg = config['preprocessing']['outliers']
    
    # Apply conditions safely if columns exist
    if 'annual_inc' in train_df.columns:
        train_df = train_df[train_df['annual_inc'] <= outlier_cfg['max_annual_inc']]
    if 'dti' in train_df.columns:
        train_df = train_df[train_df['dti'] <= outlier_cfg['max_dti']]
    if 'open_acc' in train_df.columns:
        train_df = train_df[train_df['open_acc'] <= outlier_cfg['max_open_acc']]
    if 'total_acc' in train_df.columns:
        train_df = train_df[train_df['total_acc'] <= outlier_cfg['max_total_acc']]
    if 'revol_util' in train_df.columns:
        train_df = train_df[train_df['revol_util'] <= outlier_cfg['max_revol_util']]
    if 'revol_bal' in train_df.columns:
        train_df = train_df[train_df['revol_bal'] <= outlier_cfg['max_revol_bal']]
        
    return train_df

def prepare_datasets(df, config):
    """
    Complete split, outlier filter, and scaling pipeline.
    """
    test_size = config['preprocessing']['test_size']
    random_state = config['preprocessing']['random_state']
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Filter outliers ONLY on train dataset
    train_df = filter_outliers(train_df, config)
    
    # Separate features and target
    X_train = train_df.drop('loan_status', axis=1)
    y_train = train_df['loan_status']
    X_test = test_df.drop('loan_status', axis=1)
    y_test = test_df['loan_status']
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert datasets to float32 NumPy arrays for TensorFlow compatibility
    X_train_scaled = np.array(X_train_scaled).astype(np.float32)
    X_test_scaled = np.array(X_test_scaled).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler
