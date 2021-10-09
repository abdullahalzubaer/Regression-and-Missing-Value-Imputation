import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    r"DATASET_LOCATION"
)

# Keep only these features below - "converteComp" will be the outcome
df_new = df[
    [
        "Hobbyist",
        "Age",
        "Age1stCode",
        "YearsCodePro",
        "Country",
        "EdLevel",
        "Employment",
        "ConvertedComp",
    ]
]
df_new = df_new.dropna(how="all")  # drop rows where all elements are missing
df_new = df_new.rename({"ConvertedComp": "Salary"}, axis=1)  # Renaming columns
df_new = df_new.dropna(
    subset=["Salary"]
)  # Dropping the columns that does not have the salary.

print(df_new.isnull().sum())  # Missing value in each feature
print(df_new.nunique())  # Unique values in each feature
# Getting all the unique values for Hobbyist features only.
print(f"Hobbist: {df_new['Hobbyist'].unique()}")
# Getting all the unique values for all the features.
print(pd.Series({col: df_new[col].unique() for col in df_new}))

print(
    df_new["Country"].value_counts()
)  # Countries and it's frequencey present in the dataset
# lets remove countries that are present less than 100 times in the dataset
df_new = df_new.groupby("Country").filter(lambda x: len(x) > 100)
print(df_new.isnull().sum())

# Till now we got all the features and the labels in the dataframe.

# Converting columns containing non-numerical values to numerical.
"""
We have to be careful here since we do not encode the missing values.

"""

print(f"YearsCodePro: {df_new['YearsCodePro'].unique()}")


def clean_experience(x):
    if x == "Less than 1 year":
        return 1
    elif x == "More than 50 years":
        return 50
    else:
        return float(x)


df_new["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
print(f"YearsCodePro: {df_new['YearsCodePro'].unique()}")


print(
    f"Hobbist: {df_new['Hobbyist'].unique()}"
)  # Getting all the unique values for Hobbyist features only.


def hobby(x):
    if x == "Yes":
        return 1
    elif x == "No":
        return 0
    else:
        return float(x)


df_new["Hobbyist"] = df["Hobbyist"].apply(hobby)
print(f"Hobbist: {df_new['Hobbyist'].unique()}")

print(f"Age1stCode: {df_new['Age1stCode'].unique()}")


def clean_age_first_code(x):
    if x == "Younger than 5 years":
        return 5
    elif x == "Older than 85":
        return 85
    elif isinstance(x, str):
        return float(x)
    else:
        pass


df_new["Age1stCode"] = df_new["Age1stCode"].apply(clean_age_first_code)
print(f"Age1stCode: {df_new['Age1stCode'].unique()}")

print(f"Country: {df_new['Country'].unique()}")
le_country = LabelEncoder()
df_new["Country"] = le_country.fit_transform(df_new["Country"])
print(f"Country: {df_new['Country'].unique()}")

encoders = dict()  # Will be used as the dictionary that has the encoders.

print(f"EdLevel: {df_new['EdLevel'].unique()}")
series = df_new["EdLevel"]
label_encoder = LabelEncoder()
df_new["EdLevel"] = pd.Series(
    label_encoder.fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index,
)
encoders["EdLevel"] = label_encoder
print(f"EdLevel: {df_new['EdLevel'].unique()}")

print(f"Employment: {df_new['Employment'].unique()}")
series = df_new["Employment"]
label_encoder = LabelEncoder()
df_new["Employment"] = pd.Series(
    label_encoder.fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index,
)
encoders["Employment"] = label_encoder
print(f"Employment: {df_new['Employment'].unique()}")

df_new.info()
df_new.isnull().sum()

# Imputation of the missing values.

imputer = KNNImputer(
    n_neighbors=10
)  # n_neighbors is a hyperparameter that should be tuned.

df_new_non_imputed = df_new
df_new_imputed = pd.DataFrame(imputer.fit_transform(df_new), columns=df_new.columns)
print(df_new_imputed.isnull().sum())

# Creating x and y from the imputer dataframe
X = df_new_imputed.drop("Salary", axis="columns")  # axis = 1
Y = df_new_imputed["Salary"]


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)
random_forest_regressor = RandomForestRegressor(random_state=0)
random_forest_regressor.fit(X_train, y_train)
y_predict = random_forest_regressor.predict(X_test)
error = mean_absolute_error(y_true=y_test, y_pred=y_predict)
print((error))

df_new_non_imputed_null_dropped = df_new_non_imputed.dropna(
    axis=0
)  # dropping the records with missing value_counts

X = df_new_non_imputed_null_dropped.drop("Salary", axis="columns")  # axis = 1
Y = df_new_non_imputed_null_dropped["Salary"]


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)
random_forest_regressor = RandomForestRegressor(random_state=0)
random_forest_regressor.fit(X_train, y_train)
y_predict = random_forest_regressor.predict(X_test)
error = mean_absolute_error(y_true=y_test, y_pred=y_predict)
print((error))
