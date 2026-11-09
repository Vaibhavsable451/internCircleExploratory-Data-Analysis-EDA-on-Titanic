# Exploratory Data Analysis on Titanic

Pandas · 891 passengers · cleaning, feature engineering, groupby analysis

## Key insights

**38.4%**overall survival rate

**74% vs 19%**women vs men survived

**63% / 47% / 24%**survival in 1st / 2nd / 3rd class

**97% vs 13.5%**1st-class women vs 3rd-class men

**58%**of children survived (seniors: 23%)

**51% vs 30%**with family vs alone

## Data cleaning

| Column | Missing | Fix |
| --- | --- | --- |
| Age | 177 (19.9%) | Median by Sex + Pclass group |
| Cabin | 687 (77.1%) | Replaced with HasCabin flag, dropped column |
| Embarked | 2 (0.2%) | Filled with mode (S) |
| Duplicates | 0 | Checked rows and PassengerId |

## Plots

EDA plots

## Code (titanic_eda.py)

```
"""Exploratory Data Analysis on the Titanic dataset using Pandas."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------
# 1. LOAD + BASIC DATAFRAME OPERATIONS
# ------------------------------------------------------------------
df = pd.read_csv("titanic.csv")

section("1. HEAD")
print(df.head())

section("1. INFO")
df.info()

section("1. DESCRIBE (numeric)")
print(df.describe().round(2))

section("1. DESCRIBE (categorical)")
print(df.describe(include="object"))

# ------------------------------------------------------------------
# 2. MISSING VALUES + DUPLICATES
# ------------------------------------------------------------------
section("2. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
print(pd.DataFrame({"missing": missing, "percent": missing_pct})
      .query("missing > 0"))

section("2. DUPLICATES")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate PassengerIds:", df["PassengerId"].duplicated().sum())
df = df.drop_duplicates()

# Cleaning strategy:
#  - Age (~20% missing): fill with median Age of same Sex + Pclass group
#  - Embarked (2 missing): fill with the mode
#  - Cabin (~77% missing): too sparse -> keep only a 'HasCabin' flag, drop column
df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
    lambda s: s.fillna(s.median())
)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["HasCabin"] = df["Cabin"].notna().astype(int)
df = df.drop(columns=["Cabin"])

section("2. MISSING AFTER CLEANING")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().any().any()
      else "No missing values remain.")

# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------------
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["AgeGroup"] = pd.cut(
    df["Age"], bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["Title"] = df["Title"].replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)
common = df["Title"].value_counts()
df["Title"] = df["Title"].where(df["Title"].isin(common[common >= 10].index), "Rare")

# ------------------------------------------------------------------
# 4. GROUPING + AGGREGATION
# ------------------------------------------------------------------
section("4. OVERALL SURVIVAL RATE")
print(f"{df['Survived'].mean():.1%}")

for col in ["Sex", "Pclass", "Embarked", "AgeGroup", "IsAlone", "Title"]:
    section(f"4. SURVIVAL BY {col.upper()}")
    print(
        df.groupby(col, observed=True)["Survived"]
        .agg(passengers="count", survivors="sum", survival_rate="mean")
        .round(3)
    )

section("4. SURVIVAL BY SEX x CLASS (pivot)")
print(df.pivot_table(values="Survived", index="Sex", columns="Pclass",
                     aggfunc="mean").round(3))

section("4. MULTI-AGGREGATION BY CLASS")
print(
    df.groupby("Pclass").agg(
        avg_age=("Age", "mean"),
        avg_fare=("Fare", "mean"),
        median_fare=("Fare", "median"),
        avg_family=("FamilySize", "mean"),
        survival_rate=("Survived", "mean"),
    ).round(2)
)

section("4. CORRELATION WITH SURVIVAL")
num = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare",
          "FamilySize", "HasCabin"]]
print(num.corr()["Survived"].drop("Survived").sort_values().round(3))

# ------------------------------------------------------------------
# 5. VISUALS
# ------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Titanic EDA", fontsize=16)

df.groupby("Sex")["Survived"].mean().plot.bar(ax=ax[0, 0], color=["#e07a9b", "#4a90c2"])
ax[0, 0].set_title("Survival rate by Sex")

df.groupby("Pclass")["Survived"].mean().plot.bar(ax=ax[0, 1], color="#5aa469")
ax[0, 1].set_title("Survival rate by Class")

df.pivot_table(values="Survived", index="Pclass", columns="Sex").plot.bar(ax=ax[0, 2])
ax[0, 2].set_title("Survival: Class x Sex")

df[df.Survived == 0]["Age"].plot.hist(ax=ax[1, 0], alpha=0.6, bins=20, label="Died")
df[df.Survived == 1]["Age"].plot.hist(ax=ax[1, 0], alpha=0.6, bins=20, label="Survived")
ax[1, 0].set_title("Age distribution"); ax[1, 0].legend()

df.groupby("FamilySize")["Survived"].mean().plot.bar(ax=ax[1, 1], color="#c9923b")
ax[1, 1].set_title("Survival by Family Size")

df.boxplot(column="Fare", by="Pclass", ax=ax[1, 2])
ax[1, 2].set_title("Fare by Class"); ax[1, 2].set_ylim(0, 300)
plt.suptitle("Titanic EDA", fontsize=16)

plt.tight_layout()
plt.savefig("titanic_eda_plots.png", dpi=110)

df.to_csv("titanic_cleaned.csv", index=False)
print("\nSaved: titanic_eda_plots.png, titanic_cleaned.csv")
```

## Full output

```
======================================================================
1. HEAD
======================================================================
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch  \
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0   
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0   
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0   
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0   
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0   

             Ticket     Fare Cabin Embarked  
0         A/5 21171   7.2500   NaN        S  
1          PC 17599  71.2833   C85        C  
2  STON/O2. 3101282   7.9250   NaN        S  
3            113803  53.1000  C123        S  
4            373450   8.0500   NaN        S  

======================================================================
1. INFO
======================================================================
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    str    
 4   Sex          891 non-null    str    
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    str    
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    str    
 11  Embarked     889 non-null    str    
dtypes: float64(2), int64(5), str(5)
memory usage: 83.7 KB

======================================================================
1. DESCRIBE (numeric)
======================================================================
       PassengerId  Survived  Pclass     Age   SibSp   Parch    Fare
count       891.00    891.00  891.00  714.00  891.00  891.00  891.00
mean        446.00      0.38    2.31   29.70    0.52    0.38   32.20
std         257.35      0.49    0.84   14.53    1.10    0.81   49.69
min           1.00      0.00    1.00    0.42    0.00    0.00    0.00
25%         223.50      0.00    2.00   20.12    0.00    0.00    7.91
50%         446.00      0.00    3.00   28.00    0.00    0.00   14.45
75%         668.50      1.00    3.00   38.00    1.00    0.00   31.00
max         891.00      1.00    3.00   80.00    8.00    6.00  512.33

======================================================================
1. DESCRIBE (categorical)
======================================================================
/mnt/user-data/outputs/titanic_eda.py:32: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  print(df.describe(include="object"))
                           Name   Sex  Ticket Cabin Embarked
count                       891   891     891   204      889
unique                      891     2     681   147        3
top     Braund, Mr. Owen Harris  male  347082    G6        S
freq                          1   577       7     4      644

======================================================================
2. MISSING VALUES
======================================================================
          missing  percent
Age           177    19.87
Cabin         687    77.10
Embarked        2     0.22

======================================================================
2. DUPLICATES
======================================================================
Duplicate rows: 0
Duplicate PassengerIds: 0

======================================================================
2. MISSING AFTER CLEANING
======================================================================
No missing values remain.

======================================================================
4. OVERALL SURVIVAL RATE
======================================================================
38.4%

======================================================================
4. SURVIVAL BY SEX
======================================================================
        passengers  survivors  survival_rate
Sex                                         
female         314        233          0.742
male           577        109          0.189

======================================================================
4. SURVIVAL BY PCLASS
======================================================================
        passengers  survivors  survival_rate
Pclass                                      
1              216        136          0.630
2              184         87          0.473
3              491        119          0.242

======================================================================
4. SURVIVAL BY EMBARKED
======================================================================
          passengers  survivors  survival_rate
Embarked                                      
C                168         93          0.554
Q                 77         30          0.390
S                646        219          0.339

======================================================================
4. SURVIVAL BY AGEGROUP
======================================================================
            passengers  survivors  survival_rate
AgeGroup                                        
Child               69         40          0.580
Teen                70         30          0.429
YoungAdult         514        184          0.358
Adult              216         83          0.384
Senior              22          5          0.227

======================================================================
4. SURVIVAL BY ISALONE
======================================================================
         passengers  survivors  survival_rate
IsAlone                                      
0               354        179          0.506
1               537        163          0.304

======================================================================
4. SURVIVAL BY TITLE
======================================================================
        passengers  survivors  survival_rate
Title                                       
Master          40         23          0.575
Miss           185        130          0.703
Mr             517         81          0.157
Mrs            126        100          0.794
Rare            23          8          0.348

======================================================================
4. SURVIVAL BY SEX x CLASS (pivot)
======================================================================
Pclass      1      2      3
Sex                        
female  0.968  0.921  0.500
male    0.369  0.157  0.135

======================================================================
4. MULTI-AGGREGATION BY CLASS
======================================================================
        avg_age  avg_fare  median_fare  avg_family  survival_rate
Pclass                                                           
1         38.27     84.15        60.29        1.77           0.63
2         29.86     20.66        14.25        1.78           0.47
3         24.80     13.68         8.05        2.01           0.24

======================================================================
4. CORRELATION WITH SURVIVAL
======================================================================
Pclass       -0.338
Age          -0.060
SibSp        -0.035
FamilySize    0.017
Parch         0.082
Fare          0.257
HasCabin      0.317
Name: Survived, dtype: float64

Saved: titanic_eda_plots.png, titanic_cleaned.csv
```