import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def popularized(df):
    n=len(df)

    def add_Prop(df, newcol, col, n):
        df[newcol] = df[col].count()/n
        return df

    for feature in df.column:
        pop_feature = feature + "-Prop"
        df = df.groupby(col_popularize).apply(add_Prop, pop_feature, feature, n)
    return df

def get_dummies(df):
    df = df.to_frame()
    for feature in df.columns:
        # Create a set of dummy variables from the sex variable
        df_dummies = pd.get_dummies(df)

        # Join the dummy variables to the main dataframe
        df_new = pd.concat([df, df_dummies], axis=1)
        df_new.drop([feature], axis=1)
    return df_new


# Retourne un df des variables explicatives : X
def preprocessed(df):
    copie_df = df
    del copie_df['MT_VNT_EUR_VOY']
    return copie_df

# Retourne la variable cible : y
def target(df):
    return df['MT_VNT_EUR_VOY']


def sample_data(df):
    sample_df = df.sample(n=100000)
    return sample_df