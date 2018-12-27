import requests
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def get_broad_category_icd9_l1(diag_code):
    if (diag_code >= 390 and diag_code < 460) or (np.floor(diag_code) == 785):
        return 1
    elif (diag_code >= 460 and diag_code < 520) or (np.floor(diag_code) == 786):
        return 2
    elif (diag_code >= 520 and diag_code < 580) or (np.floor(diag_code) == 787):
        return 3
    elif (np.floor(diag_code) == 250):
        return 4
    elif (diag_code >= 800 and diag_code < 1000):
        return 5
    elif (diag_code >= 710 and diag_code < 740):
        return 6
    elif (diag_code >= 580 and diag_code < 630) or (np.floor(diag_code) == 788):
        return 7
    elif (diag_code >= 140 and diag_code < 240):
        return 8
    else:
        return 0

def get_broad_category_icd9_l2(diag_code):
    if (diag_code >= 390 and diag_code < 399):
        return 1
    elif (diag_code >= 401 and diag_code < 415):
        return 2
    elif (diag_code >= 415 and diag_code < 460):
        return 3
    elif (np.floor(diag_code) == 785):
        return 4
    elif (diag_code >= 460 and diag_code < 489):
        return 5
    elif (diag_code >= 490 and diag_code < 497):
        return 6
    elif (diag_code >= 500 and diag_code < 520):
        return 7
    elif (np.floor(diag_code) == 786):
        return 8
    elif (diag_code >= 520 and diag_code < 530):
        return 9
    elif (diag_code >= 530 and diag_code < 544):
        return 10
    elif (diag_code >= 550 and diag_code < 554):
        return 11
    elif (diag_code >= 555 and diag_code < 580):
        return 12
    elif (np.floor(diag_code) == 787):
        return 13
    elif (np.floor(diag_code) == 250):
        return 14
    elif (diag_code >= 800 and diag_code < 1000):
        return 15
    elif (diag_code >= 710 and diag_code < 740):
        return 16
    elif (diag_code >= 580 and diag_code < 630):
        return 17
    elif (np.floor(diag_code) == 788):
        return 18
    elif (diag_code >= 140 and diag_code < 240):
        return 19
    elif diag_code >= 240 and diag_code < 280 and (np.floor(diag_code) != 250):
        return 20
    elif (diag_code >= 680 and diag_code < 710) or (np.floor(diag_code) == 782):
        return 21
    elif (diag_code >= 290 and diag_code < 320):
        return 22
    else:
        return 0


def icd10_to_9_convert(icd_10_diag_code):
    URL = "https://icd.codes/api?f=icdcm_10_to_9&code={}".format(icd_10_diag_code)
    result = requests.get(url = URL)
    data = result.json()
    try:
       icd_9_code = data['results'][0]['icd9v1_code']
       if icd_9_code != None:
           if len(icd_9_code) > 3:
               return float(icd_9_code[:3] + '.' + icd_9_code[3:])
       else:
           return 0
       return float(icd_9_code)
    except KeyError:
        return 0

def get_age_category(age):
    if age >= 0 and age < 10:
        return 1
    elif age >=10 and age < 20:
        return 2
    elif age >=20 and age < 30:
        return 3
    elif age >=30 and age < 40:
        return 4
    elif age >=40 and age < 50:
        return 5
    elif age >=50 and age < 60:
        return 6
    elif age >=60 and age < 70:
        return 7
    elif age >=70 and age < 80:
        return 8
    elif age >=80 and age < 90:
        return 9
    elif age >=90 and age < 100 or age > 100:
        return 10

def get_max_glu_serum_val(max_glu_serum):
    if max_glu_serum != None:
        max_glu_val = float(max_glu_serum)
    else:
        return -99

    if max_glu_val <= 180:
        return 0
    elif max_glu_val > 180:
        return 1

def get_A1Cresult_val(A1Cresult):
    if A1Cresult != None:
        hb_a1c_val = float(A1Cresult)
    else:
        return -99

    if hb_a1c_val <= 5.6:
        return 0
    elif hb_a1c_val > 5.6:
        return 1

def convert_raw_data(data_dict):
    df_pred_dict = {}

    data_dict['discharge_disposition_id'] = int(data_dict['discharge_disposition_id'])
    data_dict['admission_source_id'] = int(data_dict['admission_source_id'])
    data_dict['admission_type_id'] = int(data_dict['admission_type_id'])

    data_dict['number_outpatient'] = int(data_dict['number_outpatient'])
    data_dict['number_inpatient'] = int(data_dict['number_inpatient'])
    data_dict['number_emergency'] = int(data_dict['number_emergency'])
    data_dict['gender'] = int(data_dict['gender'])

    df_pred_dict['age'] = get_age_category(int(data_dict['age']))
    df_pred_dict['time_in_hospital'] = int(data_dict['time_in_hospital'])
    df_pred_dict['num_procedures'] = int(data_dict['num_procedures'])
    df_pred_dict['num_medications'] = int(data_dict['num_medications'])
    # df_pred_dict['number_outpatient'] = int(data_dict['number_outpatient'])
    # df_pred_dict['number_inpatient'] = int(data_dict['number_inpatient'])
    # df_pred_dict['number_emergency'] = int(data_dict['number_emergency'])
    df_pred_dict['number_diagnoses'] = int(data_dict['number_diagnoses'])
    # df_pred_dict['num_lab_procedures'] = int(data_dict['num_lab_procedures'])

    # Categorize Diagnosis
    diag1_icd9 = icd10_to_9_convert(data_dict['diag_1'])
    diag2_icd9 = icd10_to_9_convert(data_dict['diag_2'])
    diag3_icd9 = icd10_to_9_convert(data_dict['diag_3'])

    diag1_l1 = get_broad_category_icd9_l1(diag1_icd9)
    diag1_l2 = get_broad_category_icd9_l2(diag1_icd9)

    diag2_l1 = get_broad_category_icd9_l1(diag2_icd9)
    diag2_l2 = get_broad_category_icd9_l2(diag2_icd9)

    diag3_l1 = get_broad_category_icd9_l1(diag3_icd9)
    diag3_l2 = get_broad_category_icd9_l2(diag3_icd9)

    # df_pred_dict['diag_1'] = diag1_icd9
    # df_pred_dict['diag_2'] = diag2_icd9
    # df_pred_dict['diag_3'] = diag3_icd9

    # df_pred_dict['level1_diag1'] = diag1_l1
    # df_pred_dict['level2_diag1'] = diag1_l2
    # df_pred_dict['level1_diag2'] = diag2_l1
    # df_pred_dict['level2_diag2'] = diag2_l2
    # df_pred_dict['level1_diag3'] = diag3_l1
    # df_pred_dict['level2_diag3'] = diag3_l2

    df_pred_dict['level1_diag1_1.0'] = 0
    df_pred_dict['level1_diag1_2.0'] = 0
    df_pred_dict['level1_diag1_3.0'] = 0
    df_pred_dict['level1_diag1_4.0'] = 0
    df_pred_dict['level1_diag1_5.0'] = 0
    df_pred_dict['level1_diag1_6.0'] = 0
    df_pred_dict['level1_diag1_7.0'] = 0
    df_pred_dict['level1_diag1_8.0'] = 0

    if diag1_l1 in (1,2,3,4,5,6,7,8):
        diag_key = 'level1_diag1_{}'.format(float(diag1_l1))
        df_pred_dict[diag_key] = 1


    # Shorten categories
    if data_dict['discharge_disposition_id'] in (6,9,8,13):
        data_dict['discharge_disposition_id'] = 1
    elif data_dict['discharge_disposition_id'] in (3,4,5,14,22,23,24):
        data_dict['discharge_disposition_id'] = 2
    elif data_dict['discharge_disposition_id'] in (12,15,16,17):
        data_dict['discharge_disposition_id'] = 10
    elif data_dict['discharge_disposition_id'] in (25,26):
        data_dict['discharge_disposition_id'] = 18


    if data_dict['admission_type_id'] in (2,7):
        data_dict['admission_type_id'] = 1
    elif data_dict['admission_type_id'] in (6,8):
        data_dict['admission_type_id'] = 5

    if data_dict['admission_source_id'] in (2,3):
        data_dict['admission_source_id'] = 1
    elif data_dict['admission_source_id'] in (5,6,10,22,25):
        data_dict['admission_source_id'] = 4
    elif data_dict['admission_source_id'] in (15,17,20,21):
        data_dict['admission_source_id'] = 9
    elif data_dict['admission_source_id'] in (13,14):
        data_dict['admission_source_id'] = 11



    med_list = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

    race_keys = ['race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other']

    for race in race_keys:
        if data_dict['race'] in race:
            df_pred_dict[race] = 1
        else:
            df_pred_dict[race] = 0

    if data_dict['gender'] == 1:
        df_pred_dict['gender_1'] = data_dict['gender']
    else:
        df_pred_dict['gender_1'] = 0

    for med in med_list:
        df_pred_dict[med] = data_dict[med]

    df_pred_dict['number_outpatient_log1p'] = np.log1p(data_dict['number_outpatient'])
    df_pred_dict['number_emergency_log1p'] = np.log1p(data_dict['number_emergency'])
    df_pred_dict['number_inpatient_log1p'] = np.log1p(data_dict['number_inpatient'])

    df_pred_dict['admission_type_id_3'] = 0
    df_pred_dict['admission_type_id_5'] = 0

    df_pred_dict['discharge_disposition_id_2'] = 0
    df_pred_dict['discharge_disposition_id_7'] = 0
    df_pred_dict['discharge_disposition_id_10'] = 0
    df_pred_dict['discharge_disposition_id_18'] = 0

    df_pred_dict['admission_source_id_4'] = 0
    df_pred_dict['admission_source_id_7'] = 0
    df_pred_dict['admission_source_id_9'] = 0

    if data_dict['admission_type_id'] in (3, 5):
        df_pred_dict['admission_type_id_'+str(data_dict['admission_type_id'])] = 1

    if data_dict['discharge_disposition_id'] in (2,7,10,18):
        df_pred_dict['discharge_disposition_id_'+str(data_dict['discharge_disposition_id'])] = 1

    if data_dict['admission_source_id'] in (4,7,9):
        df_pred_dict['admission_source_id_'+str(data_dict['admission_source_id'])] = 1

    df_pred_dict['max_glu_serum_0'] = get_max_glu_serum_val(data_dict['max_glu_serum'])
    df_pred_dict['max_glu_serum_1'] = 0
    df_pred_dict['A1Cresult_0'] = get_A1Cresult_val(data_dict['A1Cresult'])
    df_pred_dict['A1Cresult_1'] = 0

    return df_pred_dict

def run_prediction(pickle_model_name, X_in):
    loaded_model = pickle.load(open(pickle_model_name, 'rb'))
    readmission_predict = loaded_model.predict(X_in)
    predict_probability = loaded_model.predict_proba(X_in)
    return {
        "prediction_value": str(readmission_predict[0]),
        "probability_0": str(predict_probability[:,0][0]),
        "probability_1": str(predict_probability[:,1][0])
    }

def start_prediction(data_payload):
    payload_dict = convert_raw_data(data_payload)

    feature_list = ['age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient_log1p',
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
    np_list = []
    for feature in feature_list:
        val = (payload_dict[feature])
        np_list.append(val)

    pred_val = run_prediction('finalized_model.pkl', np.array([np_list]))
    # print(pred_val)
    return pred_val

if __name__ == '__main__':
    data = {
        'discharge_disposition_id': '1',
        'admission_source_id': '4',
        'admission_type_id': '1',
        'number_outpatient': '0',
        'number_inpatient': '0',
        'number_emergency': '0',
        'age': '65',
        'time_in_hospital': '7',
        'num_procedures': '0',
        'num_lab_procedures': '0',
        'num_medications': '11',
        'number_diagnoses': '7',
        'race': 'AfricanAmerican',
        'gender': '1',
        'max_glu_serum': None,
        'A1Cresult': None,
        'diag_1': 'C25.0',
        'diag_2': 'E10.11',
        'diag_3': 'J98.5',
        'metformin': '0',
        'repaglinide': '0',
        'nateglinide': '0',
        'chlorpropamide': '0',
        'glimepiride': '0',
        'acetohexamide': '0',
        'glipizide': '0',
        'glyburide': '1',
        'tolbutamide': '0',
        'pioglitazone': '0',
        'rosiglitazone': '0',
        'acarbose': '0',
        'miglitol': '0',
        'troglitazone': '0',
        'tolazamide': '0',
        'insulin': '1',
        'glyburide-metformin': '0',
        'glipizide-metformin': '0',
        'glimepiride-pioglitazone': '0',
        'metformin-rosiglitazone': '0',
        'metformin-pioglitazone': '0'
    }
    start_prediction(data)
