import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# SEASON 2022-23
bib_url_2023 = 'https://www.basketball-reference.com/leagues/NBA_2023.html'
requests.get(bib_url_2023)
data_2023=pd.read_html(bib_url_2023)[5]
data2_2023=pd.read_html(bib_url_2023)[10]
df = pd.DataFrame(data_2023)
data2_2023.columns = data2_2023.columns.droplevel(0)
merged_data_2023 = data_2023.merge(data2_2023[["Team", "Age", "MOV", "Pace", "TS%", "W", "L"]], on='Team', how='left')
merged_data_2023['Team'] = merged_data_2023['Team'].str.replace('*', '', regex=False)
merged_data_2023
merged_data_2023['Year'] = 2023
# Drop the last row from merged_data
merged_data_2023 = merged_data_2023.drop(merged_data_2023.index[-1])

# SEASON 2021-22
bib_url_2022 = 'https://www.basketball-reference.com/leagues/NBA_2022.html'
requests.get(bib_url_2022)
data_2022=pd.read_html(bib_url_2022)[5]
data2_2022=pd.read_html(bib_url_2022)[10]
df = pd.DataFrame(data_2022)
data2_2022.columns = data2_2022.columns.droplevel(0)
merged_data_2022 = data_2022.merge(data2_2022[["Team", "Age", "MOV", "Pace", "TS%", "W", "L"]], on='Team', how='left')
merged_data_2022['Team'] = merged_data_2022['Team'].str.replace('*', '', regex=False)
merged_data_2022
merged_data_2022['Year'] = 2022
# Drop the last row from merged_data
merged_data_2022 = merged_data_2022.drop(merged_data_2022.index[-1])

# SEASON 2020-21
bib_url_2021 = 'https://www.basketball-reference.com/leagues/NBA_2021.html'
requests.get(bib_url_2021)
data_2021=pd.read_html(bib_url_2021)[5]
data2_2021=pd.read_html(bib_url_2021)[10]
df = pd.DataFrame(data_2021)
data2_2021.columns = data2_2021.columns.droplevel(0)
merged_data_2021 = data_2021.merge(data2_2021[["Team", "Age", "MOV", "Pace", "TS%", "W", "L"]], on='Team', how='left')
merged_data_2021['Team'] = merged_data_2021['Team'].str.replace('*', '', regex=False)
merged_data_2021
merged_data_2021['Year'] = 2021
# Drop the last row from merged_data
merged_data_2021 = merged_data_2021.drop(merged_data_2021.index[-1])

# SEASON 2019-20
bib_url_2020 = 'https://www.basketball-reference.com/leagues/NBA_2020.html'
requests.get(bib_url_2020)
data_2020=pd.read_html(bib_url_2020)[5]
data2_2020=pd.read_html(bib_url_2020)[10]
df = pd.DataFrame(data_2020)
data2_2020.columns = data2_2020.columns.droplevel(0)
merged_data_2020 = data_2020.merge(data2_2020[["Team", "Age", "MOV", "Pace", "TS%", "W", "L"]], on='Team', how='left')
merged_data_2020['Team'] = merged_data_2020['Team'].str.replace('*', '', regex=False)
merged_data_2020
merged_data_2020['Year'] = 2020
# Drop the last row from merged_data
merged_data_2020 = merged_data_2020.drop(merged_data_2020.index[-1])

# SEASON 2018-19
bib_url_2019 = 'https://www.basketball-reference.com/leagues/NBA_2019.html'
requests.get(bib_url_2019)
data_2019=pd.read_html(bib_url_2019)[5]
data2_2019=pd.read_html(bib_url_2019)[10]
df = pd.DataFrame(data_2019)
data2_2019.columns = data2_2019.columns.droplevel(0)
merged_data_2019 = data_2019.merge(data2_2019[["Team", "Age", "MOV", "Pace", "TS%", "W", "L"]], on='Team', how='left')
merged_data_2019['Team'] = merged_data_2019['Team'].str.replace('*', '', regex=False)
merged_data_2019
merged_data_2019['Year'] = 2019
# Drop the last row from merged_data
merged_data_2019 = merged_data_2019.drop(merged_data_2019.index[-1])

# SEASON 2017-18
bib_url_2018 = 'https://www.basketball-reference.com/leagues/NBA_2018.html'
requests.get(bib_url_2018)
data_2018=pd.read_html(bib_url_2018)[5]
data2_2018=pd.read_html(bib_url_2018)[10]
df = pd.DataFrame(data_2018)
data2_2018.columns = data2_2018.columns.droplevel(0)
merged_data_2018 = data_2018.merge(data2_2018[["Team", "Age", "MOV", "Pace", "TS%", "W", "L"]], on='Team', how='left')
merged_data_2018['Team'] = merged_data_2018['Team'].str.replace('*', '', regex=False)
merged_data_2018
merged_data_2018['Year'] = 2018
# Drop the last row from merged_data
merged_data_2018 = merged_data_2018.drop(merged_data_2018.index[-1])

# Concatenate all data frames into one
all_data = pd.concat([merged_data_2023, merged_data_2022, merged_data_2021,
                     merged_data_2020, merged_data_2019, merged_data_2018])



#run this after editing all_data_1
all_data_1 = all_data