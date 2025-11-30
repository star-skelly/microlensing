import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load your data
fdir = "data/generated_lightcurves/xy/"
directory = os.fsencode(fdir)
files = [file.decode("utf-8") for file in os.listdir(directory)]
df = pd.DataFrame(data=files, columns=["file_num"])

# First split: train vs (val+test)
train_ids, valtest_ids = train_test_split(
    files,
    test_size=0.70,    # 30% goes to val+test
    random_state=42
)

# Second split: validation vs test
val_ids, test_ids = train_test_split(
    valtest_ids,
    test_size=0.50,    # half goes to test, half to val
    random_state=42
)

# Now filter the dataframe based on hotspot_id
train_df = df[df["file_num"].isin(train_ids)]
val_df   = df[df["file_num"].isin(val_ids)]
test_df  = df[df["file_num"].isin(test_ids)]

# Save them
train_df.to_csv("data/generated_lightcurves/train.csv", index=False)
val_df.to_csv("data/generated_lightcurves/val.csv", index=False)
test_df.to_csv("data/generated_lightcurves/test.csv", index=False)

print("Done! Saved train.csv, val.csv, test.csv")
