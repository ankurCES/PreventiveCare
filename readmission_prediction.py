# import libraries needed
import IPython
import requests
import pandas as pd
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import seaborn as sns
import pickle
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def standardize(raw_data):
    return ((raw_data - np.mean(raw_data, axis = 0)) / np.std(raw_data, axis = 0))

def run_modelling():
    df = pd.read_csv("diabetes_data_preprocessed.csv", index_col=0)

    # convert data type of nominal features in dataframe to 'object' type
    i = ['encounter_id', 'patient_nbr', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose','miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'age', 'A1Cresult', 'max_glu_serum', 'level1_diag1', 'level1_diag2', 'level1_diag3', 'level2_diag1', 'level2_diag2', 'level2_diag3']

    df[i] = df[i].astype('object')

    L1 = np.random.randint(1,10,20)
    L2 = np.random.randint(1,20, 20)

    datframe = pd.DataFrame()
    datframe['L1'] = L1
    datframe['L2'] = L2

    datframe.corr()

    scaler = MinMaxScaler()
    datframe = pd.DataFrame(scaler.fit_transform(datframe), columns = ['L1', 'L2'])
    datframe.corr()

    df['age'] = df['age'].astype('int64')
    print(df.age.value_counts())
    # age categories -> mp vals
    age_dict = {1:5, 2:15, 3:25, 4:35, 5:45, 6:55, 7:65, 8:75, 9:85, 10:95}
    df['age'] = df.age.map(age_dict)
    print(df.age.value_counts())

    num_col = list(set(list(df._get_numeric_data().columns))- {'readmitted'})

    # Log transform
    statdataframe = pd.DataFrame()
    statdataframe['numeric_column'] = num_col
    skew_before = []
    skew_after = []

    kurt_before = []
    kurt_after = []

    standard_deviation_before = []
    standard_deviation_after = []

    log_transform_needed = []

    log_type = []

    for i in num_col:
        skewval = df[i].skew()
        skew_before.append(skewval)

        kurtval = df[i].kurtosis()
        kurt_before.append(kurtval)

        sdval = df[i].std()
        standard_deviation_before.append(sdval)

        if (abs(skewval) >2) & (abs(kurtval) >2):
            log_transform_needed.append('Yes')

            if len(df[df[i] == 0])/len(df) <=0.02:
                log_type.append('log')
                skewvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).skew()
                skew_after.append(skewvalnew)

                kurtvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).kurtosis()
                kurt_after.append(kurtvalnew)

                sdvalnew = np.log(pd.DataFrame(df[train_data[i] > 0])[i]).std()
                standard_deviation_after.append(sdvalnew)

            else:
                log_type.append('log1p')
                skewvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).skew()
                skew_after.append(skewvalnew)

                kurtvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).kurtosis()
                kurt_after.append(kurtvalnew)

                sdvalnew = np.log1p(pd.DataFrame(df[df[i] >= 0])[i]).std()
                standard_deviation_after.append(sdvalnew)

        else:
            log_type.append('NA')
            log_transform_needed.append('No')

            skew_after.append(skewval)
            kurt_after.append(kurtval)
            standard_deviation_after.append(sdval)

    statdataframe['skew_before'] = skew_before
    statdataframe['kurtosis_before'] = kurt_before
    statdataframe['standard_deviation_before'] = standard_deviation_before
    statdataframe['log_transform_needed'] = log_transform_needed
    statdataframe['log_type'] = log_type
    statdataframe['skew_after'] = skew_after
    statdataframe['kurtosis_after'] = kurt_after
    statdataframe['standard_deviation_after'] = standard_deviation_after

    for i in range(len(statdataframe)):
        if statdataframe['log_transform_needed'][i] == 'Yes':
            colname = str(statdataframe['numeric_column'][i])

            if statdataframe['log_type'][i] == 'log':
                df = df[df[colname] > 0]
                df[colname + "_log"] = np.log(df[colname])

            elif statdataframe['log_type'][i] == 'log1p':
                df = df[df[colname] >= 0]
                df[colname + "_log1p"] = np.log1p(df[colname])

    df = df.drop(['number_outpatient', 'number_inpatient', 'number_emergency','service_utilization'], axis = 1)

    # get list of only numeric features
    numerics = list(set(list(df._get_numeric_data().columns))- {'readmitted'})

    df.encounter_id = df.encounter_id.astype('int64')
    df.patient_nbr = df.patient_nbr.astype('int64')
    df.diabetesMed = df.diabetesMed.astype('int64')
    df.change = df.change.astype('int64')

    # convert data type of nominal features in dataframe to 'object' type for aggregating
    i = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose','miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone','A1Cresult']
    df[i] = df[i].astype('int64')

    df.dtypes

    df.A1Cresult.value_counts()

    dfcopy = df.copy(deep = True)
    df = dfcopy.copy(deep = True)

    df['readmitted'] = df['readmitted'].apply(lambda x: 0 if x == 2 else x)

    # drop individual diagnosis columns that have too granular disease information
    # drop level 2 categorization (which was not comparable with any reference)
    # drop level 1 secondary and tertiary diagnoses
    df.drop(['diag_1', 'diag_2', 'diag_3', 'level2_diag1', 'level1_diag2', 'level2_diag2', 'level1_diag3', 'level2_diag3'], axis=1, inplace=True)

    df.head(2)

    interactionterms = [
        ('num_medications','time_in_hospital'),
        ('num_medications','num_procedures'),
        ('time_in_hospital','num_lab_procedures'),
        ('num_medications','num_lab_procedures'),
        ('num_medications','number_diagnoses'),
        ('age','number_diagnoses'),
        ('change','num_medications'),
        ('number_diagnoses','time_in_hospital'),
        ('num_medications','numchange')
    ]

    for inter in interactionterms:
        name = inter[0] + '|' + inter[1]
        df[name] = df[inter[0]] * df[inter[1]]


    df[['num_medications','time_in_hospital', 'num_medications|time_in_hospital']].head()

    datf = pd.DataFrame()
    datf['features'] = numerics
    datf['std_dev'] = datf['features'].apply(lambda x: df[x].std())
    datf['mean'] = datf['features'].apply(lambda x: df[x].mean())
    # duplicate enc removal
    df2 = df.drop_duplicates(subset= ['patient_nbr'], keep = 'first')
    df2.shape

    df2[numerics] = standardize(df2[numerics])

    df2 = df2[(np.abs(sp.stats.zscore(df2[numerics])) < 3).all(axis=1)]

    my_cmap = ListedColormap(sns.light_palette((250, 100, 50), input="husl", n_colors=50).as_hex())
    table = df2.drop(['patient_nbr', 'encounter_id'], axis=1).corr(method='pearson')
    table.style.background_gradient(cmap=my_cmap, axis = 0)

    pd.options.display.max_rows = 400

    c = df2.corr().abs()
    s = c.unstack()
    print(s.shape)
    so = s.sort_values(ascending=False)

    df2['level1_diag1'] = df2['level1_diag1'].astype('object')
    df_pd = pd.get_dummies(df2, columns=['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'max_glu_serum', 'A1Cresult', 'level1_diag1'], drop_first = True)

    non_num_cols = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'max_glu_serum', 'A1Cresult', 'level1_diag1' ]

    num_cols = list(set(list(df._get_numeric_data().columns))- {'readmitted', 'change'})

    new_non_num_cols = []
    for i in non_num_cols:
        for j in df_pd.columns:
            if i in j:
                new_non_num_cols.append(j)


    l = []
    for feature in list(df_pd.columns):
        if '|' in feature:
            l.append(feature)


    # Create Model

    feature_set_1_no_int = ['age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient_log1p',
                     'number_emergency_log1p', 'number_inpatient_log1p', 'number_diagnoses', 'metformin',
                     'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide',
                     'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                     'tolazamide', 'insulin', 'glyburide-metformin',
                     'race_AfricanAmerican', 'race_Asian', 'race_Caucasian',
                     'race_Hispanic', 'race_Other', 'gender_1',
                     'admission_type_id_3', 'admission_type_id_5',
                     'discharge_disposition_id_2', 'discharge_disposition_id_7',
                     'discharge_disposition_id_10', 'discharge_disposition_id_18',
                     'admission_source_id_4', 'admission_source_id_7',
                     'admission_source_id_9', 'max_glu_serum_0',
                     'max_glu_serum_1', 'A1Cresult_0', 'A1Cresult_1',
                     'level1_diag1_1.0',
                     'level1_diag1_2.0',
                     'level1_diag1_3.0',
                     'level1_diag1_4.0',
                     'level1_diag1_5.0',
                     'level1_diag1_6.0',
                     'level1_diag1_7.0',
                     'level1_diag1_8.0']

    train_input = df_pd[feature_set_1_no_int]
    train_output = df_pd['readmitted']

    # Data balancing applied
    print('Original dataset shape {}'.format(Counter(train_output)))
    smt = SMOTE(random_state=20)
    train_input_new, train_output_new = smt.fit_sample(train_input, train_output)
    print('New dataset shape {}'.format(Counter(train_output_new)))
    train_input_new = pd.DataFrame(train_input_new, columns = list(train_input.columns))
    X_train, X_dev, Y_train, Y_dev = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)

    # forrest = RandomForestClassifier(n_estimators = 10, max_depth=25, criterion = "entropy", min_samples_split=10)
    # print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(forrest, X_train, Y_train, cv=10))))
    # forrest.fit(X_train, Y_train)
    # print("Dev Set score: {:.2%}".format(forrest.score(X_dev, Y_dev)))
    #
    # Y_dev_predict = forrest.predict(X_dev)
    # pd.crosstab(pd.Series(Y_dev, name = 'Actual'), pd.Series(Y_dev_predict, name = 'Predict'), margins = True)
    #
    # print("Accuracy is {0:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)))
    # print("Precision is {0:.2f}".format(precision_score(Y_dev, Y_dev_predict)))
    # print("Recall is {0:.2f}".format(recall_score(Y_dev, Y_dev_predict)))
    # print("AUC is {0:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict)))
    #
    # accuracy_forreste = accuracy_score(Y_dev, Y_dev_predict)
    # precision_forreste = precision_score(Y_dev, Y_dev_predict)
    # recall_forreste = recall_score(Y_dev, Y_dev_predict)
    # auc_forreste = roc_auc_score(Y_dev, Y_dev_predict)

    # Create list of top most features based on importance
    # feature_names = X_train.columns
    # feature_imports = forrest.feature_importances_
    # most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
    # most_imp_features.sort_values(by="Importance", inplace=True)
    # plt.figure(figsize=(10,6))
    # plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
    # plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
    # plt.xlabel('Importance')
    # plt.title('Most important features - Random Forest (Gini function, complex model)')
    # plt.savefig('comp_model_features_1.jpg')

    forrest = RandomForestClassifier(n_estimators = 10, max_depth=300, criterion = 'entropy', random_state = 0, min_samples_split=10)
    print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(forrest, X_train, Y_train, cv=10))))
    forrest.fit(X_train, Y_train)
    print("Dev Set score: {:.2%}".format(forrest.score(X_dev, Y_dev)))

    Y_dev_predict = forrest.predict(X_dev)

    # print(Y_dev_predict)
    # print('================')
    # print(Y_dev)

    pd.crosstab(pd.Series(Y_dev, name = 'Actual'), pd.Series(Y_dev_predict, name = 'Predict'), margins = True)

    # print("Accuracy is {0:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)))
    # print("Precision is {0:.2f}".format(precision_score(Y_dev, Y_dev_predict)))
    # print("Recall is {0:.2f}".format(recall_score(Y_dev, Y_dev_predict)))
    # print("AUC is {0:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict)))

    accuracy_forrestg = accuracy_score(Y_dev, Y_dev_predict)
    precision_forrestg = precision_score(Y_dev, Y_dev_predict)
    recall_forrestg = recall_score(Y_dev, Y_dev_predict)
    auc_forrestg = roc_auc_score(Y_dev, Y_dev_predict)

    score_dict = {
        "Accuracy" : "{0:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)),
        "Precision" : "{0:.2f}".format(precision_score(Y_dev, Y_dev_predict)),
        "Recall": "{0:.2f}".format(recall_score(Y_dev, Y_dev_predict)),
        "AUC": "{0:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict))
    }

    pickle_model_name = 'finalized_model.pkl'
    pickle.dump(forrest, open(pickle_model_name, 'wb'))

    print("Model pickle file {} saved!!".format(pickle_model_name))

    # Create list of top most features based on importance
    feature_names = X_train.columns
    feature_imports = forrest.feature_importances_
    most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
    most_imp_features.sort_values(by="Importance", inplace=True)
    plt.figure(figsize=(10,6))
    plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
    plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
    plt.xlabel('Importance')
    plt.title('Most important features - Random Forest (gini) (2 - complex model)')
    plt.savefig('comp_model_features_2.jpg')
    # run_prediction(pickle_model_name, X_dev, Y_dev)
    return score_dict


def run_prediction(pickle_model_name, X_dev, Y_dev):
    loaded_model = pickle.load(open(pickle_model_name, 'rb'))
    result = loaded_model.score(X_dev, Y_dev)
    # print("Result : {}".format(result))
    # print(X_train.columns)

if __name__ == '__main__':
    run_modelling()
