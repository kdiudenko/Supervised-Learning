import pandas as pd
from numpy import log2


def entropy(m, n):
    t = m + n
    return -(n/t)*log2(n/t) - (m/t)*log2(m/t)


def entropy_species_value_counts(df):
    m = df["Mobug"]
    n = df["Lobug"]
    return entropy(m, n), m + n


def calc_gain(en_df, en_l, en_r, m, n):
    return en_df - ((m / (m + n)) * en_l + (n / (m + n)) * en_r)


df = pd.read_csv('ml-bugs.csv')

n = len(df[df.Species == 'Mobug'])
m = len(df[df.Species == 'Lobug'])
total = len(df)


en_df = entropy(m, n)

en_color_brown, m = entropy_species_value_counts(df[df.Color == 'Brown'].Species.value_counts())
en_color_no_brown, n = entropy_species_value_counts(df[df.Color != 'Brown'].Species.value_counts())
gain_color_brown = calc_gain(en_df, en_color_brown, en_color_no_brown, m, n)
print(f"{gain_color_brown=}")

en_color_blue, m = entropy_species_value_counts(df[df.Color == 'Blue'].Species.value_counts())
en_color_no_blue, n = entropy_species_value_counts(df[df.Color != 'Blue'].Species.value_counts())
gain_color_blue = calc_gain(en_df, en_color_blue, en_color_no_blue, m, n)
print(f"{gain_color_blue=}")

en_color_green, m = entropy_species_value_counts(df[df.Color == 'Green'].Species.value_counts())
en_color_no_green, n = entropy_species_value_counts(df[df.Color != 'Green'].Species.value_counts())
gain_color_green = calc_gain(en_df, en_color_green, en_color_no_green, m, n)
print(f"{gain_color_green=}")

en_length_17, m = entropy_species_value_counts(df[df["Length (mm)"] < 17.0].Species.value_counts())
en_color_no_length_17, n = entropy_species_value_counts(df[df["Length (mm)"] >= 17.0].Species.value_counts())
gain_length_17 = calc_gain(en_df, en_length_17, en_color_no_length_17, m, n)
print(f"{gain_length_17=}")

en_length_20, m = entropy_species_value_counts(df[df.iloc[:,2] < 20.0].Species.value_counts())
en_color_no_length_20, n = entropy_species_value_counts(df[df.iloc[:,2] >= 20.0].Species.value_counts())
gain_length_20 = calc_gain(en_df, en_length_20, en_color_no_length_20, m, n)
print(f"{gain_length_20=}")