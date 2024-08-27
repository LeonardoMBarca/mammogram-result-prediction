import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

file = "data/raw/mammographic_masses.data"
col_names = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']

df = pd.read_csv(file, na_values=['?'], names=col_names)
df.drop("BI-RADS", axis=1, inplace=True)

df.dropna(subset=["Age", "Shape", "Margin"], inplace=True)

x_train = df[df["Density"].notnull()]
x_train.drop("Density", axis=1, inplace=True)
y_train = df[df["Density"].notnull()]["Density"]
x_test = df[df["Density"].isnull()]
x_test.drop("Density", axis=1, inplace=True)
y_test = df[df["Density"].isnull()]["Density"]

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
df.loc[df["Density"].isnull(), "Density"] = y_predict

columns_to_scale = ['Age', 'Density', 'Margin', 'Shape']

scaler = StandardScaler()
scaler.fit(df[columns_to_scale])
scaled_columns = scaler.transform(df[columns_to_scale])

scaled_df = pd.DataFrame(scaled_columns, columns=columns_to_scale, index=df.index)
df[columns_to_scale] = scaled_df

output = "data/processed/processed_data.csv"
df.to_csv(output, index=False)