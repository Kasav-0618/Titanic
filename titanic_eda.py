"""
================================================================================
TITANIC EXPLORATORY DATA ANALYSIS (EDA) & DATA CLEANING
Target Learning Objectives:
  1. Core DataFrame operations (head, info, describe, filtering, selection)
  2. Handling missing values and duplicate records
  3. Grouping and aggregation analysis (groupby, agg, pivot_table, crosstab)
================================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def main():
    # -------------------------------------------------------------------------
    # 0. Load Dataset
    # -------------------------------------------------------------------------
    csv_file = "titanic.csv"
    if not os.path.exists(csv_file):
        alt_paths = ["../titanic.csv", "train.csv"]
        for p in alt_paths:
            if os.path.exists(p):
                csv_file = p
                break
    
    print(f"[INFO] Reading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)

    # =========================================================================
    # REQUIREMENT 1: DATAFRAME OPERATIONS (head, info, describe)
    # =========================================================================
    section("REQUIREMENT 1: DATAFRAME OPERATIONS")

    # 1.1 head() and tail()
    print("\n--- 1.1 First 5 Rows (df.head()) ---")
    print(df.head())

    print("\n--- 1.2 Last 3 Rows (df.tail(3)) ---")
    print(df.tail(3))

    # 1.3 Dimensions & Column Types
    print("\n--- 1.3 DataFrame Shape (df.shape) ---")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    print("\n--- 1.4 Detailed Schema & Memory Usage (df.info()) ---")
    df.info()

    # 1.5 Summary Statistics: Numerical
    print("\n--- 1.5 Numerical Summary Statistics (df.describe()) ---")
    print(df.describe().round(2))

    # 1.6 Summary Statistics: Categorical
    print("\n--- 1.6 Categorical Summary Statistics (df.describe(include=['O'])) ---")
    print(df.describe(include=['O']))

    # 1.7 Conditional Selection & Slicing
    print("\n--- 1.7 Conditional Slicing (First Class Female Survivors) ---")
    first_class_women = df[(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Survived'] == 1)]
    print(f"Found {len(first_class_women)} passengers matching criteria. Sample:")
    print(first_class_women[['PassengerId', 'Name', 'Age', 'Fare']].head(3))

    # =========================================================================
    # REQUIREMENT 2: HANDLING MISSING VALUES AND DUPLICATES
    # =========================================================================
    section("REQUIREMENT 2: HANDLING MISSING VALUES AND DUPLICATES")

    # 2.1 Duplicate Detection
    print("\n--- 2.1 Duplicate Rows Check (df.duplicated()) ---")
    num_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows in dataset: {num_duplicates}")
    if num_duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed duplicates. New shape: {df.shape}")
    else:
        print("Dataset contains zero identical duplicate rows.")

    # 2.2 Missing Values Identification
    print("\n--- 2.2 Missing Values Identification ---")
    null_counts = df.isnull().sum()
    null_percent = (df.isnull().sum() / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': null_counts,
        'Percentage (%)': null_percent.round(2)
    })
    print(missing_report[missing_report['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False))

    # 2.3 Data Cleaning & Imputation Pipeline
    print("\n--- 2.3 Cleaning & Imputation Pipeline ---")
    df_clean = df.copy()

    # Strategy A: Embarked (Categorical Mode Imputation)
    embarked_mode = df_clean['Embarked'].mode()[0]
    print(f"A. Imputing 'Embarked' missing values with mode: '{embarked_mode}'")
    df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)

    # Strategy B: Age (Contextual Median Imputation using Title and Pclass)
    # Extract titles (e.g. Mr, Mrs, Miss, Master)
    df_clean['Title'] = df_clean['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
    # Group rare titles
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df_clean['Title'] = df_clean['Title'].replace(rare_titles, 'Rare')
    df_clean['Title'] = df_clean['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    
    print("B. Imputing 'Age' with median grouped by Title and Pclass:")
    age_medians = df_clean.groupby(['Pclass', 'Title'])['Age'].transform('median')
    # Fallback to general Pclass median if combination is missing
    df_clean['Age'] = df_clean['Age'].fillna(age_medians)
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

    # Strategy C: Cabin (Deck Extraction & Explicit Missing Category)
    print("C. Processing 'Cabin': Creating 'Deck' feature and 'HasCabin' indicator")
    df_clean['Deck'] = df_clean['Cabin'].str[0].fillna('Unknown')
    df_clean['HasCabin'] = df_clean['Cabin'].notnull().astype(int)

    # Strategy D: Fare Outlier & Missing Handling
    df_clean['Fare'] = df_clean['Fare'].fillna(df_clean.groupby('Pclass')['Fare'].transform('median'))
    df_clean['LogFare'] = np.log1p(df_clean['Fare'])

    print("\n--- 2.4 Post-Cleaning Verification ---")
    remaining_nulls = df_clean.isnull().sum().sum()
    print(f"Total remaining null values across cleaned dataframe: {remaining_nulls}")
    print(f"Cleaned DataFrame Shape: {df_clean.shape}")

    # =========================================================================
    # REQUIREMENT 3: GROUPING AND AGGREGATION ANALYSIS
    # =========================================================================
    section("REQUIREMENT 3: GROUPING AND AGGREGATION ANALYSIS")

    # 3.1 Single-column Groupby: Survival by Gender
    print("\n--- 3.1 GroupBy: Survival Rate & Counts by Sex ---")
    sex_analysis = df_clean.groupby('Sex')['Survived'].agg(
        Total_Passengers='count',
        Total_Survivors='sum',
        Survival_Rate='mean'
    )
    sex_analysis['Survival_Rate (%)'] = (sex_analysis['Survival_Rate'] * 100).round(2)
    print(sex_analysis[['Total_Passengers', 'Total_Survivors', 'Survival_Rate (%)']])

    # 3.2 Single-column Groupby: Socioeconomic Class Metrics
    print("\n--- 3.2 GroupBy: Multi-metric Aggregations across Pclass ---")
    pclass_agg = df_clean.groupby('Pclass').agg(
        Passenger_Count=('PassengerId', 'count'),
        Survival_Rate=('Survived', 'mean'),
        Average_Age=('Age', 'mean'),
        Median_Fare=('Fare', 'median'),
        Max_Fare=('Fare', 'max')
    ).round(2)
    pclass_agg['Survival_Rate'] = (pclass_agg['Survival_Rate'] * 100).round(2)
    print(pclass_agg)

    # 3.3 Multi-column Groupby: Sex and Pclass Interaction
    print("\n--- 3.3 Multi-column GroupBy: [Pclass, Sex] Hierarchy ---")
    interaction_df = df_clean.groupby(['Pclass', 'Sex'])['Survived'].agg(
        Count='count',
        Survived_Count='sum',
        Survival_Pct=lambda x: round(x.mean() * 100, 2)
    )
    print(interaction_df)

    # 3.4 Pivot Tables (pd.pivot_table)
    print("\n--- 3.4 Pivot Table: Mean Survival by Pclass (rows) and Sex (columns) ---")
    pivot_survival = pd.pivot_table(
        df_clean,
        values='Survived',
        index='Pclass',
        columns='Sex',
        aggfunc='mean',
        margins=True,
        margins_name='Overall'
    )
    print((pivot_survival * 100).round(2))

    # 3.5 Cross-Tabulation (pd.crosstab)
    print("\n--- 3.5 Cross-Tabulation: Embarkation Port vs Pclass ---")
    crosstab_emb = pd.crosstab(
        df_clean['Embarked'], 
        df_clean['Pclass'], 
        margins=True, 
        margins_name='Total'
    )
    print(crosstab_emb)

    # 3.6 Advanced Derived Aggregations: Family Dynamics
    print("\n--- 3.6 Family Dynamics & Survival ---")
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)

    family_agg = df_clean.groupby('FamilySize')['Survived'].agg(
        Count='count',
        Survival_Rate=lambda x: f"{x.mean()*100:.1f}%"
    )
    print(family_agg)

    # Save cleaned dataset for modeling/further analysis
    output_cleaned = "titanic_cleaned.csv"
    df_clean.to_csv(output_cleaned, index=False)
    print(f"\n[SUCCESS] Cleaned dataset successfully saved to: {output_cleaned}")

if __name__ == "__main__":
    main()
