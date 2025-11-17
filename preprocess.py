import pandas as pd
import numpy as np
import os

# Define paths
data_dir = r'E:\SEM 4\INT256\datasets'
processed_dir = r'E:\SEM 4\INT256\processed_data'
os.makedirs(processed_dir, exist_ok=True)

# Paths to the UNSW-NB15 dataset files
data_path = data_dir  # Updated to point directly to datasets folder
train_file = os.path.join(data_path, 'UNSW_NB15_training-set.csv')
test_file = os.path.join(data_path, 'UNSW_NB15_testing-set.csv')

# Define the columns we need
usecols = ['sbytes', 'dbytes', 'rate', 'label', 'attack_cat']
dtypes = {
    'sbytes': 'float32',
    'dbytes': 'float32',
    'rate': 'float32',
    'label': 'int32',
    'attack_cat': 'object'
}

# Function to normalize data
def normalize_data(df):
    for feature in ['sbytes', 'dbytes', 'rate']:
        min_val = df[feature].min()
        max_val = df[feature].max()
        if max_val > min_val:
            df[feature] = (df[feature] - min_val) / (max_val - min_val)
        else:
            df[feature] = 0
    return df

# Process the training data in chunks
print("Processing training data...")
train_chunks = pd.read_csv(train_file, usecols=usecols, dtype=dtypes, chunksize=10000)

train_normal_chunks = []
train_attack_chunks = []

for chunk in train_chunks:
    # Split normal and attack data
    normal_chunk = chunk[chunk['label'] == 0].copy()
    attack_chunk = chunk[chunk['label'] == 1].copy()
    
    # Normalize normal data to [0, 1]
    if not normal_chunk.empty:
        normal_chunk = normalize_data(normal_chunk)
        train_normal_chunks.append(normal_chunk)
    
    # Normalize attack data to [1, 2]
    if not attack_chunk.empty:
        attack_chunk = normalize_data(attack_chunk)
        attack_chunk[['sbytes', 'dbytes', 'rate']] = attack_chunk[['sbytes', 'dbytes', 'rate']] + 1
        train_attack_chunks.append(attack_chunk)

# Concatenate chunks
train_normal = pd.concat(train_normal_chunks, ignore_index=True) if train_normal_chunks else pd.DataFrame(columns=usecols)
train_attack = pd.concat(train_attack_chunks, ignore_index=True) if train_attack_chunks else pd.DataFrame(columns=usecols)

# Process the test data in chunks
print("Processing test data...")
test_chunks = pd.read_csv(test_file, usecols=usecols, dtype=dtypes, chunksize=10000)

test_normal_chunks = []
test_attack_chunks = []

for chunk in test_chunks:
    # Split normal and attack data
    normal_chunk = chunk[chunk['label'] == 0].copy()
    attack_chunk = chunk[chunk['label'] == 1].copy()
    
    # Normalize normal data to [0, 1]
    if not normal_chunk.empty:
        normal_chunk = normalize_data(normal_chunk)
        test_normal_chunks.append(normal_chunk)
    
    # Normalize attack data to [1, 2]
    if not attack_chunk.empty:
        attack_chunk = normalize_data(attack_chunk)
        attack_chunk[['sbytes', 'dbytes', 'rate']] = attack_chunk[['sbytes', 'dbytes', 'rate']] + 1
        test_attack_chunks.append(attack_chunk)

# Concatenate chunks
test_normal = pd.concat(test_normal_chunks, ignore_index=True) if test_normal_chunks else pd.DataFrame(columns=usecols)
test_attack = pd.concat(test_attack_chunks, ignore_index=True) if test_attack_chunks else pd.DataFrame(columns=usecols)

# Combine test normal and attack data
test_data = pd.concat([test_normal, test_attack], ignore_index=True)

# Categorize attacks into Malware and DDoS
# Malware: Worms, Backdoor, Shellcode
# DDoS: DoS, DDoS
malware_categories = ['Worms', 'Backdoor', 'Shellcode']
ddos_categories = ['DoS', 'DDoS']

train_attack_malware = train_attack[train_attack['attack_cat'].isin(malware_categories)]
train_attack_ddos = train_attack[train_attack['attack_cat'].isin(ddos_categories)]

# Drop the attack_cat column from the final data
train_normal = train_normal.drop(columns=['attack_cat'])
train_attack_malware = train_attack_malware.drop(columns=['attack_cat'])
train_attack_ddos = train_attack_ddos.drop(columns=['attack_cat'])
test_data = test_data.drop(columns=['attack_cat'])

# Rename columns to match the expected format in ais_threat_detection.py
train_normal.rename(columns={'sbytes': 'src_bytes', 'dbytes': 'dst_bytes'}, inplace=True)
train_attack_malware.rename(columns={'sbytes': 'src_bytes', 'dbytes': 'dst_bytes'}, inplace=True)
train_attack_ddos.rename(columns={'sbytes': 'src_bytes', 'dbytes': 'dst_bytes'}, inplace=True)
test_data.rename(columns={'sbytes': 'src_bytes', 'dbytes': 'dst_bytes'}, inplace=True)

# Save processed data
train_normal.to_csv(os.path.join(processed_dir, 'normal_data.csv'), index=False)
train_attack_malware.to_csv(os.path.join(processed_dir, 'threat_malware.csv'), index=False)
train_attack_ddos.to_csv(os.path.join(processed_dir, 'threat_ddos.csv'), index=False)
test_data.to_csv(os.path.join(processed_dir, 'test_data.csv'), index=False)

print("Data preprocessing completed. Files saved in processed_data folder.")