import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def prepare_modeling_sets(df, time_column='Timestamp'):
    # 1. Sort by Time to prevent Information Leakage
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # 2. Define Features and Target
    X = df.drop(['Label', 'Label_Category', time_column], axis=1)
    y = df['Label_Category']
    
    # 3. Chronological Split (No Shuffling)
    # We take the first 80% for training/val and last 20% for final testing
    split_index = int(len(df) * 0.8)
    X_train_val, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train_val, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Split the train_val into train (75%) and validation (25%)
    # effectively giving a 60/20/20 split of total data
    val_split_index = int(len(X_train_val) * 0.75)
    X_train, X_val = X_train_val.iloc[:val_split_index], X_train_val.iloc[val_split_index:]
    y_train, y_val = y_train_val.iloc[:val_split_index], y_train_val.iloc[val_split_index:]
    
    # 4. Robust Scaling
    # We fit the scaler ONLY on training data to prevent leakage
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train size: {X_train_scaled.shape}")
    print(f"Validation size: {X_val_scaled.shape}")
    print(f"Test size: {X_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

# Note: Ensure all non-numeric columns (IPs, Ports) are encoded or dropped before scaling
