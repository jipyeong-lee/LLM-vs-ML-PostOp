from config.settings import set_environment_variables
from data.load_data import load_data
from data.preprocess import impute_data
from models.setup_llm import setup_llm
from utils.prompts import generate_prompts
from utils.helpers import convert_to_text, convert_to_json, extract_log_probs, softmax
import pandas as pd
import pickle
from tqdm import tqdm
from vllm import SamplingParams

def run():
    """Main execution function"""
    # 1. Set environment variables
    set_environment_variables()

    # 2. Load data
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()

    # 3. Preprocess data (handle missing values)
    x_train_imputed, x_valid_imputed, x_test_imputed = impute_data(x_train, x_valid, x_test)

    # 4. Convert to text and JSON format
    x_train['text'] = x_train_imputed.apply(lambda x: convert_to_text(x), axis=1)
    x_valid['text'] = x_valid_imputed.apply(lambda x: convert_to_text(x), axis=1)
    x_test['text'] = x_test_imputed.apply(lambda x: convert_to_text(x), axis=1)

    x_train['json'] = x_train_imputed.apply(lambda x: convert_to_json(x), axis=1)
    x_valid['json'] = x_valid_imputed.apply(lambda x: convert_to_json(x), axis=1)
    x_test['json'] = x_test_imputed.apply(lambda x: convert_to_json(x), axis=1)

    # 5. Set up the LLM model
    llm = setup_llm()

    # 6. Generate prompts
    first_sentence = """
    This patient is 75.0 years old, female. Requiring emergency operation. The Body Mass Index (BMI) is 20.811654526534856. Preoperative tests show glucose levels at 110.0 milligrams per deciliter, hemoglobin at 11.0 grams per deciliter, albumin levels at 3.6 grams per deciliter, creatinine levels at 5.55 milligrams per deciliter, Blood Urea Nitrogen (BUN) at 50.0 milligrams per deciliter, platelet count at 200.0 times 10 to the power of 9 per liter, white blood cell (WBC) count at 7.9 times 10 to the power of 9 per liter, Activated Partial Thromboplastin Time (APTT) at 29.6 seconds, Prothrombin Time International Normalized Ratio (PT INR) at 0.9, sodium at 132.0 millimoles per liter, potassium at 5.3 millimoles per liter, Aspartate Aminotransferase (AST) at 19.0 Units per Liter, Alanine Aminotransferase (ALT) at 11.0 Units per Liter, and anesthesia duration was 200.0 minutes.
    """.strip()
    first_reason = """
The prediction of {"inhosp_death_30day":"True"} for the given patient is based on the following risk factors:
- Age (75 years): Advanced age is a significant risk factor for increased mortality due to reduced physiological reserves and higher likelihood of comorbidities.
- Emergency Operation: Emergency surgeries are associated with higher risk since there is limited time for preoperative optimization, often reflecting a critical underlying condition.
- Hemoglobin (11.0 g/dL): A low-normal hemoglobin level may suggest mild anemia, which can exacerbate perioperative complications, especially in elderly patients.
- Anesthesia Duration (200 minutes): Longer anesthesia times are correlated with more complex procedures, increasing the likelihood of adverse outcomes.
- Blood Urea Nitrogen (BUN) (50.0 mg/dL): Elevated BUN levels indicate significant renal dysfunction, a strong predictor of postoperative mortality.
- Creatinine (5.55 mg/dL): Severely elevated creatinine suggests advanced renal impairment, which complicates perioperative management and increases the risk of poor outcomes.
- Sodium (132.0 mmol/L): Hyponatremia (low sodium) is associated with an increased risk of neurological complications and mortality.
- Potassium (5.3 mmol/L): Hyperkalemia (high potassium) poses a risk for life-threatening cardiac arrhythmias, particularly in the perioperative period.
- Albumin (3.6 g/dL): Although within the normal range, it is on the lower end, reflecting a borderline nutritional or inflammatory status.
Based on the combination of these risk factors, the cumulative risk for 30-day postoperative mortality is high, leading to the prediction of {"inhosp_death_30day":"True"}.
    """.strip()

    second_sentence = """
    This patient is 55.0 years old, female. Requiring emergency operation. The Body Mass Index (BMI) is 23.4375. Preoperative tests show glucose levels at 120.0 milligrams per deciliter, hemoglobin at 11.9 grams per deciliter, albumin levels at 3.9 grams per deciliter, creatinine levels at 0.58 milligrams per deciliter, Blood Urea Nitrogen (BUN) at 16.0 milligrams per deciliter, platelet count at 233.0 times 10 to the power of 9 per liter, white blood cell (WBC) count at 13.79 times 10 to the power of 9 per liter, Activated Partial Thromboplastin Time (APTT) at 25.1 seconds, Prothrombin Time International Normalized Ratio (PT INR) at 0.95, sodium at 143.0 millimoles per liter, potassium at 3.8 millimoles per liter, Aspartate Aminotransferase (AST) at 11.0 Units per Liter, Alanine Aminotransferase (ALT) at 13.0 Units per Liter, and anesthesia duration was 240.0 minutes.
    """.strip()
    second_reason = """
The prediction of {"inhosp_death_30day":"False"} for the given patient is based on the following considerations:
- Age (55 years): The patient is middle-aged, which is associated with lower risk compared to elderly patients who typically have reduced physiological reserves and a higher prevalence of comorbidities.
- Emergency Operation: While emergency surgeries increase the risk of mortality, other factors in this patient's profile mitigate the overall risk.
- Hemoglobin (11.9 g/dL): This hemoglobin level is close to normal and does not indicate significant anemia, reducing the risk of perioperative complications.
- Anesthesia Duration (240 minutes): Although the surgery duration is long, it alone does not indicate a high risk due to the patient's overall stable condition.
- White Blood Cell Count (13.79 × 10⁹/L): Elevated WBC count suggests systemic inflammation or a potential infection, but this is not severe enough in isolation to indicate a high mortality risk.
- Blood Urea Nitrogen (BUN) (16.0 mg/dL): The BUN level is within normal range, indicating good renal function and reducing the risk associated with renal dysfunction.
- Creatinine (0.58 mg/dL): Normal creatinine levels indicate adequate renal function, lowering the risk of complications related to impaired kidney function.
- Sodium (143.0 mmol/L): Sodium levels are normal, reducing the risk of neurological complications and mortality linked to sodium imbalances.
- Potassium (3.8 mmol/L): Normal potassium levels indicate no immediate risk of cardiac arrhythmias due to electrolyte imbalance.
- Albumin (3.9 g/dL): The albumin level is within normal range, reflecting good nutritional status and the ability to respond to surgical stress.
- Additional Stable Laboratory Values: Platelet count, aPTT, PT-INR, AST, and ALT are all within normal ranges, further suggesting no significant risk from bleeding, coagulation disorders, or liver dysfunction.
Based on the overall profile, the patient's preoperative condition does not indicate a high risk of 30-day postoperative mortality, supporting the prediction of {"inhosp_death_30day":"False"}.
    """.strip()

    dataset_valid = pd.DataFrame(x_valid)
    dataset_valid = generate_prompts(dataset_valid, first_sentence, first_reason, second_sentence, second_reason)

    dataset_test = pd.DataFrame(x_test)
    dataset_test = generate_prompts(dataset_test, first_sentence, first_reason, second_sentence, second_reason)

    # 7. Model inference
    sampling_params = SamplingParams(
        max_tokens=30,
        temperature=0.0,
        skip_special_tokens=True,
        logprobs=10,
    )

    all_prompts_valid_sentence = dataset_valid['2shot_false_prompt'].tolist()
    all_prompts_test_sentence = dataset_test['2shot_false_prompt'].tolist()

    all_outputs_valid_sentence = llm.generate(all_prompts_valid_sentence, sampling_params)
    all_outputs_test_sentence = llm.generate(all_prompts_test_sentence, sampling_params)

    outputs_valid_2shot_false = all_outputs_valid_sentence[-len(dataset_valid['2shot_false_prompt']):]
    outputs_test_2shot_false = all_outputs_test_sentence[-len(dataset_test['2shot_false_prompt']):]

    valid_2shot_false = []
    test_2shot_false = []

    for output in outputs_valid_2shot_false:
        result = output.outputs[0]
        valid_2shot_false.append(result)

    for output in outputs_test_2shot_false:
        result = output.outputs[0]
        test_2shot_false.append(result)

    # 8. Process and save results
    valid_2shot_false_probs = []
    test_2shot_false_probs = []

    for result in tqdm(valid_2shot_false):
        true_log_prob, false_log_prob = extract_log_probs(result.logprobs)
        log_probs = [false_log_prob, true_log_prob]
        probabilities = softmax(log_probs)
        valid_2shot_false_probs.append(probabilities[1])

    for result in tqdm(test_2shot_false):
        true_log_prob, false_log_prob = extract_log_probs(result.logprobs)
        log_probs = [false_log_prob, true_log_prob]
        probabilities = softmax(log_probs)
        test_2shot_false_probs.append(probabilities[1])

    # 9. Save results (anonymized path)
    output_path = os.getenv('OUTPUT_PATH', '/path/to/anonymous/output')
    param_type = 'BF16'
    input_type = 'sentence'
    task_type = 'inhosp_death_30day'
    
    with open(f'{output_path}/pred_2shot_false_valid_inspire.pkl', 'wb') as file:
        pickle.dump(valid_2shot_false_probs, file)

    with open(f'{output_path}/pred_2shot_false_test_inspire.pkl', 'wb') as file:
        pickle.dump(test_2shot_false_probs, file)

if __name__ == '__main__':
    run()
