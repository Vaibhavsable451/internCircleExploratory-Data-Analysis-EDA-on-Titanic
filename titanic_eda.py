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
