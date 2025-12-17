import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv("MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_acc.head()

single_file_gyro = pd.read_csv("MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

file = glob("MetaMotion/*.csv")
len(file)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

datapath = "MetaMotion/"
f = file[0]

participant = f.split('-')[0].replace(datapath,'')
label = f.split('-')[1]
category = f.split('-')[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category
print(df.head())
       
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

acc_set = 1
gyro_set = 1

for f in file:
   participant = f.split('-')[0].replace(datapath,'')
   label = f.split('-')[1]
   category = f.split('-')[2].rstrip("123").rstrip("_MetaWear_2019")
   
   df = pd.read_csv(f)
   df["participant"] = participant
   df["label"] = label
   df["category"] = category
   
   if "Accelerometer" in f:
       df["set"] = acc_set
       acc_set += 1
       acc_df = pd.concat([acc_df,df])
       
   elif "Gyroscope" in f:
       df["set"] = gyro_set
       gyro_set += 1
       gyro_df = pd.concat([gyro_df,df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()
pd.to_datetime(df["epoch (ms)"], unit="ms")

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")  
gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyro_df["epoch (ms)"]
del gyro_df["time (01:00)"]
del gyro_df["elapsed (s)"]

# -----------------------------------------------
# ---------------
# Turn into function
# --------------------------------------------------------------


file = glob("MetaMotion/*.csv")

def read_data_from_files(file):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1

    for f in file:
        participant = f.split('-')[0].replace(datapath,'')
        label = f.split('-')[1]
        category = f.split('-')[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df,df])
            
        elif "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df,df])
        
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")  
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]
    
    return acc_df, gyro_df


acc_df, gyro_df = read_data_from_files(file)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:,:3],gyro_df],axis=1)


# data_merged.dropna()

data_merged.columns = [
"acc_x",
"acc_y",
"acc_z",
"gyr_x",
"gyr_y",
"gyr_z",
"label",
"category",
"participant",
"set"
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

data_merged[: 1000].resample(rule="200ms").apply(sampling)
# Split by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("data_resampled.pkl")