import numpy as np

def convert_to_text(patient_data):
    """환자 데이터를 텍스트로 변환"""
    full_text = (
        f"This patient is {patient_data['age']} years old, "
        f"{'male' if patient_data['sex'] == 1.0 else 'female'}. "
    )
    
    if np.isnan(patient_data['emop']):
        full_text += "It is not sure that requiring emergency operation. "
    else:
        full_text += f"{'Requiring' if patient_data['emop'] else 'Not requiring'} emergency operation. "

    full_text += (
        f"The Body Mass Index (BMI) is {patient_data['bmi']}. "
        f"Preoperative tests show glucose levels at {patient_data['preop_glucose']} milligrams per deciliter, "
        f"hemoglobin at {patient_data['preop_hb']} grams per deciliter, "
        f"albumin levels at {patient_data['preop_albumin']} grams per deciliter, "
        f"creatinine levels at {patient_data['preop_creatinine']} milligrams per deciliter, "
        f"Blood Urea Nitrogen (BUN) at {patient_data['preop_bun']} milligrams per deciliter, "
        f"platelet count at {patient_data['preop_platelet']} times 10 to the power of 9 per liter, "
        f"white blood cell (WBC) count at {patient_data['preop_wbc']} times 10 to the power of 9 per liter, "
        f"Activated Partial Thromboplastin Time (APTT) at {patient_data['preop_aptt']} seconds, "
        f"Prothrombin Time International Normalized Ratio (PT INR) at {patient_data['preop_ptinr']}, "
        f"sodium at {patient_data['preop_sodium']} millimoles per liter, "
        f"potassium at {patient_data['preop_potassium']} millimoles per liter, "
        f"Aspartate Aminotransferase (AST) at {patient_data['preop_ast']} Units per Liter, "
        f"Alanine Aminotransferase (ALT) at {patient_data['preop_alt']} Units per Liter, "
        f"and anesthesia duration was {patient_data['andur']} minutes."
    )
    return ''.join(full_text)

def convert_to_json(patient_data):
    """환자 데이터를 JSON 형식으로 변환"""
    patient_json = {
        "age": patient_data["age"],
        "sex": "male" if patient_data["sex"] == 1.0 else "female",
        "bmi": patient_data["bmi"],
        "emergency_operation": 'null' if np.isnan(patient_data['emop']) else ("required" if patient_data["emop"] else "not required"),
        "preop_tests": {
            "glucose_levels": patient_data["preop_glucose"],
            "hemoglobin": patient_data["preop_hb"],
            "albumin_levels": patient_data["preop_albumin"],
            "creatinine_levels": patient_data["preop_creatinine"],
            "bun": patient_data["preop_bun"],
            "platelet_count": patient_data["preop_platelet"],
            "wbc_count": patient_data["preop_wbc"],
            "aptt": patient_data["preop_aptt"],
            "pt_inr": patient_data["preop_ptinr"]
        },
        "electrolytes": {
            "sodium": patient_data["preop_sodium"],
            "potassium": patient_data["preop_potassium"]
        },
        "liver_function": {
            "ast": patient_data["preop_ast"],
            "alt": patient_data["preop_alt"]
        },
        "anesthesia_duration": patient_data["andur"]
    }
    return patient_json

def extract_log_probs(data):
    """로그 확률 추출"""
    true_log_prob = None
    false_log_prob = None

    for token_set in data:
        for token_id, logprob_obj in token_set.items():
            if logprob_obj.decoded_token == 'True':
                true_log_prob = logprob_obj.logprob
            elif logprob_obj.decoded_token == 'False':
                false_log_prob = logprob_obj.logprob

    if true_log_prob is None:
        true_log_prob = float('-inf')
    if false_log_prob is None:
        false_log_prob = float('-inf')
                
    return true_log_prob, false_log_prob

def softmax(log_probs):
    """소프트맥스 함수"""
    probs = [math.exp(lp) for lp in log_probs]
    total = sum(probs)
    return [p / total for p in probs]