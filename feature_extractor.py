import os
import pandas as pd
import numpy as np
from sklearn import preprocessing


class FeatureExtractor(object):
    def __init__(self):
        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'), compression='zip', low_memory=False)

        award = award.dropna(subset=['incumbent_name'])
        award['incumbent_name'] = award['incumbent_name'].str.lower()
        award['incumbent_name'] = award['incumbent_name'].str.replace(r'[^\w]', '')
        award['incumbent_name'] = award['incumbent_name'].str.normalize('NFKD').str.encode('ascii',
                                                                                           errors='ignore').str.decode(
            'utf-8')

        award = award[(award['incumbent_name'] != "") & (award['incumbent_name'] != " ")]
        date = pd.to_datetime(award['Publication_date'], format='%Y-%m-%d')
        award['year'] = date.dt.year

        award['CPV_classes'] = award['CPV_classes'].str.slice(start=0, stop=3)
        award['CPV_classes'] = pd.to_numeric(award['CPV_classes'], errors='coerce')

        self.award_features = award.groupby(['incumbent_name', 'year']).agg({'CallID': pd.Series.count,
                                                                             'CPV_classes': np.min, 'amount': np.sum,
                                                                             'number_of_received_bids': np.sum,
                                                                             'Departments_of_publication': pd.Series.nunique})
        self.award_features['CPV_classes'] = self.award_features['CPV_classes'].fillna(-100)
        new_amount = self.award_features['amount'].quantile(.4)
        self.award_features['amount'] = self.award_features['amount'].replace(0., new_amount)
        self.award_features.loc[(self.award_features.amount > 100000000), 'amount'] = 100000000
        self.award_features['number_of_received_bids'] = self.award_features['number_of_received_bids'].replace(0, 1)
        self.award_features['Departments_of_publication'] = self.award_features['Departments_of_publication'].replace(0,
                                                                                                                      1)
        self.award_features = self.award_features.reset_index('year')
        self.award_features['year'] = pd.to_numeric(self.award_features['year'], errors='coerce')
        self.award_features['year'] = self.award_features['year'] - 2013
        self.award_features = self.award_features.reset_index('incumbent_name')


    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_df = X_df.drop(['Legal_ID', 'Address', 'Fiscal_year_end_date', 'Fiscal_year_duration_in_months'], axis=1)

        X_df['Name'] = X_df['Name'].str.lower()
        X_df['Name'] = X_df['Name'].str.replace(r'[^\w]', '')
        X_df['Name'] = X_df['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        X_df = X_df[(X_df['Name'] != "") & (X_df['Name'] != " ")]

        fill_na = X_df.groupby('Name')['Activity_code (APE)', 'Zipcode', 'City', 'Headcount'].last()
        fill_na.rename(columns={'Activity_code (APE)': 'Activity_code_fillna', 'Zipcode': 'Zipcode_fillna',
                                'City': 'City_fillna', 'Headcount': 'Headcount_fillna'}, inplace=True)

        X_df = pd.merge(X_df, fill_na, on='Name', how='left')

        X_df['Activity_code (APE)'] = X_df['Activity_code (APE)'].fillna(X_df['Activity_code_fillna'])
        X_df['Zipcode'] = X_df['Zipcode'].fillna(X_df['Zipcode_fillna'])
        X_df['City'] = X_df['City'].fillna(X_df['City_fillna'])
        X_df['Headcount'] = X_df['Headcount'].fillna(X_df['Headcount_fillna'])

        X_df = X_df.drop(['Activity_code_fillna', 'Zipcode_fillna', 'City_fillna', 'Headcount_fillna'], axis=1)

        X_df['Number_of_NaNs'] = X_df.isna().sum(axis=1)

        X_df['Zipcode_fillna'] = X_df['City'].str.replace(r'[^\d]', '')
        X_df['Zipcode_fillna'] = X_df['Zipcode_fillna'].replace('', '000000', inplace=True)
        X_df['Zipcode_fillna'] = X_df['Zipcode_fillna'].fillna('000000')
        X_df['Zipcode'] = X_df['Zipcode'].fillna(X_df['Zipcode_fillna'])
        X_df['Zipcode'] = X_df['Zipcode'].astype(str)
        X_df['Zipcode'] = X_df['Zipcode'].str.replace(r'[^\d]', '')
        X_df['Zipcode'] = X_df['Zipcode'].str.pad(width=5, side='right', fillchar='0')
        X_df['Departement'] = X_df['Zipcode'].apply(lambda x: int(x[0:2]))
        X_df['Zip1'] = X_df['Zipcode'].apply(lambda x: int(x[2]))
        X_df['Zip2'] = X_df['Zipcode'].apply(lambda x: int(x[3]))
        X_df['Zip3'] = X_df['Zipcode'].apply(lambda x: int(x[4]))

        X_df = X_df.drop(['Zipcode', 'Zipcode_fillna'], axis=1)
        X_df['Departement'] = pd.to_numeric(X_df['Departement'], errors='coerce')
        X_df['Zip1'] = pd.to_numeric(X_df['Zip1'], errors='coerce')
        X_df['Zip2'] = pd.to_numeric(X_df['Zip2'], errors='coerce')
        X_df['Zip3'] = pd.to_numeric(X_df['Zip3'], errors='coerce')

        X_df["Activity_code (APE)"] = X_df["Activity_code (APE)"].fillna(
            X_df["Activity_code (APE)"].mode().to_list()[0])
        le = preprocessing.LabelEncoder()
        X_df["APE4"] = le.fit_transform(X_df["Activity_code (APE)"].apply(lambda x: x[len(x) - 1]))

        X_df["Activity_code (APE)"] = X_df["Activity_code (APE)"].astype(str)
        X_df["Activity_code (APE)"] = X_df["Activity_code (APE)"].str.replace(r'[^\d]', '')
        X_df["Activity_code (APE)"] = X_df["Activity_code (APE)"].str.pad(width=4, side='right', fillchar='0')

        X_df["APE1"] = X_df["Activity_code (APE)"].apply(lambda x: int(x[0:2]))
        X_df["APE2"] = X_df["Activity_code (APE)"].apply(lambda x: int(x[2]))
        X_df["APE3"] = X_df["Activity_code (APE)"].apply(lambda x: int(x[3]))

        X_df = X_df.drop(['Activity_code (APE)'], axis=1)
        X_df['APE1'] = pd.to_numeric(X_df['APE1'], errors='coerce')
        X_df['APE2'] = pd.to_numeric(X_df['APE2'], errors='coerce')
        X_df['APE3'] = pd.to_numeric(X_df['APE3'], errors='coerce')
        X_df['APE4'] = pd.to_numeric(X_df['APE4'], errors='coerce')

        X_df["Name"].fillna("Unknown", inplace=True)
        X_df["Type_Entreprise"] = np.zeros(X_df["Name"].shape[0])
        X_df.loc[X_df['Name'].str.contains('S.A'), 'Type_Entreprise'] = 2
        X_df.loc[X_df['Name'].str.contains('SARL'), 'Type_Entreprise'] = 1
        X_df.loc[X_df['Name'].str.contains('S.A.R.L.'), 'Type_Entreprise'] = 1
        X_df.loc[X_df['Name'].str.contains('EURL'), 'Type_Entreprise'] = 3
        X_df.loc[X_df['Name'].str.contains('E.U.R.L'), 'Type_Entreprise'] = 3
        X_df.loc[X_df['Name'].str.contains('SELARL'), 'Type_Entreprise'] = 4
        X_df.loc[X_df['Name'].str.contains('S.E.L.A.R.L'), 'Type_Entreprise'] = 4
        X_df.loc[X_df['Name'].str.contains('SAS', 'S.A.S'), 'Type_Entreprise'] = 5
        X_df.loc[X_df['Name'].str.contains('S.A.S'), 'Type_Entreprise'] = 5
        X_df.loc[X_df['Name'].str.contains('SNC'), 'Type_Entreprise'] = 6
        X_df.loc[X_df['Name'].str.contains('S.N.C'), 'Type_Entreprise'] = 6
        X_df.loc[X_df['Name'].str.contains('SCP'), 'Type_Entreprise'] = 7
        X_df.loc[X_df['Name'].str.contains('S.C.P'), 'Type_Entreprise'] = 7
        X_df.loc[X_df['Name'].str.contains('RESTAURANT'), 'Type_Entreprise'] = 8
        X_df.loc[X_df['Name'].str.contains('CABINET'), 'Type_Entreprise'] = 9
        X_df.loc[X_df['Name'].str.contains('COOPERATIVE'), 'Type_Entreprise'] = 10

        X_df["Type_Entreprise"].fillna(0, inplace=True)

        X_df['Type_Entreprise'] = pd.to_numeric(X_df['Type_Entreprise'], errors='coerce')

        Mean_Headcount_SA = X_df[X_df['Name'].str.contains('S.A.')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('S.A.'), 'Headcount'].fillna(Mean_Headcount_SA, inplace=True)
        Mean_Headcount_SARL = X_df[X_df['Name'].str.contains('SARL', 'S.A.R.L')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('SARL'), 'Headcount'].fillna(Mean_Headcount_SARL, inplace=True)
        X_df.loc[X_df['Name'].str.contains('S.A.R.L'), 'Headcount'].fillna(Mean_Headcount_SARL, inplace=True)
        Mean_Headcount_EURL = X_df[X_df['Name'].str.contains('EURL', 'E.U.R.L')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('EURL'), 'Headcount'].fillna(Mean_Headcount_EURL, inplace=True)
        X_df.loc[X_df['Name'].str.contains('E.U.R.L'), 'Headcount'].fillna(Mean_Headcount_EURL, inplace=True)
        Mean_Headcount_SELARL = X_df[X_df['Name'].str.contains('SELARL', 'S.E.L.A.R.L')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('SELARL'), 'Headcount'].fillna(Mean_Headcount_SELARL, inplace=True)
        X_df.loc[X_df['Name'].str.contains('S.E.L.A.R.L'), 'Headcount'].fillna(Mean_Headcount_SELARL, inplace=True)
        Mean_Headcount_SAS = X_df[X_df['Name'].str.contains('SAS', 'S.A.S')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('SAS'), 'Headcount'].fillna(Mean_Headcount_SAS, inplace=True)
        X_df.loc[X_df['Name'].str.contains('S.A.S'), 'Headcount'].fillna(Mean_Headcount_SAS, inplace=True)
        Mean_Headcount_SNC = X_df[X_df['Name'].str.contains('SNC', 'S.N.C')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('SNC'), 'Headcount'].fillna(Mean_Headcount_SNC, inplace=True)
        X_df.loc[X_df['Name'].str.contains('S.N.C'), 'Headcount'].fillna(Mean_Headcount_SNC, inplace=True)
        Mean_Headcount_SCP = X_df[X_df['Name'].str.contains('SCP', 'S.C.P')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('SCP'), 'Headcount'].fillna(Mean_Headcount_SCP, inplace=True)
        X_df.loc[X_df['Name'].str.contains('S.C.P'), 'Headcount'].fillna(Mean_Headcount_SCP, inplace=True)
        Mean_Headcount_CABINET = X_df[X_df['Name'].str.contains('CABINET')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('CABINET'), 'Headcount'].fillna(Mean_Headcount_CABINET, inplace=True)
        Mean_Headcount_RESTAURANT = X_df[X_df['Name'].str.contains('RESTAURANT')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('RESTAURANT'), 'Headcount'].fillna(Mean_Headcount_RESTAURANT, inplace=True)
        Mean_Headcount_COOPERATIVE = X_df[X_df['Name'].str.contains('COOPERATIVE')]['Headcount'].quantile(0.5)
        X_df.loc[X_df['Name'].str.contains('COOPERATIVE'), 'Headcount'].fillna(Mean_Headcount_COOPERATIVE, inplace=True)

        X_df['City'] = X_df['City'].str.lower()
        X_df['City'] = X_df['City'].replace(r'[^\w]', '')
        X_df['City'] = X_df['City'].replace('cedex', '')
        Villes = X_df['City'].value_counts().to_dict()
        X_df['Taille_Ville'] = X_df['City'].map(Villes)
        X_df['Taille_Ville'] = X_df['Taille_Ville'].fillna(1)
        X_df['Taille_Ville'] = pd.to_numeric(X_df['Taille_Ville'], errors='coerce')

        X_df = X_df.drop(['City'], axis=1)

        X_df['Year'] = X_df['Year'] - 2012
        X_df['Year'] = X_df['Year'].fillna(0)
        X_df['Year'] = pd.to_numeric(X_df['Year'], errors='coerce')

        final_df = pd.merge(X_df, self.award_features, left_on=['Name', 'Year'], right_on=['incumbent_name', 'year'],
                            how='left')

        final_df.loc[(final_df.Headcount < 0), 'Headcount'] = np.nan
        final_df['group'] = 0
        final_df.loc[(final_df.Headcount.notna()), 'group'] = 1
        final_df['Headcount'] = final_df['Headcount'].fillna(-1)
        final_df.loc[(final_df.CPV_classes.notna()), 'group'] = 2
        final_df = final_df.fillna(-1)
        final_df = final_df.drop(['Name', 'incumbent_name', 'year'], axis=1)

        return final_df.to_numpy()