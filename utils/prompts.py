def generate_prompts(dataset, first_sentence, first_reason, second_sentence, second_reason):
    """프롬프트 생성"""
    prompt_template_2shot_false = '''
def predict_inhosp_death_30day(raw_patient_data: Union[str, dict]) -> dict:
    """
    Predict 30-day postoperative mortality risk based on preoperative data.

    This function evaluates the risk of postoperative mortality within 30 days 
    using patient demographic and preoperative laboratory data.

    **Input**:
    ----------
    `raw_patient_data` : Union[str, dict]
        - JSON-like dictionary containing demographic and lab results.
        - Text-based summary of patient data (parsed into structured format).

        **Required Fields** (for JSON input):
        - `"age"`: int, patient’s age in years.
        - `"gender"`: str, "male" or "female".
        - `"bmi"`: float, body mass index.
        - `"emergency_operation"`: bool, whether the surgery is classified as emergency.
        - `"preoperative_lab_results"`: dict, including:
            - `"hemoglobin"`, `"platelet_count"`, `"wbc"`, `"aPTT"`, `"glucose"`,
              `"bun"`, `"albumin"`, `"ast"`, `"alt"`, `"creatinine"`, `"sodium"`, `"potassium"`.

    **Few-Shot Examples**:
    ----------------------
    Example 1:
    Input:
    ```json
    {second_data}
    ```
    Output:
    ```json
    {{
        "inhosp_death_30day":"False",
        "explanation":{second_reason}
    }}
    ```

    Example 2:
    Input:
    ```json
    {first_data}
    ```
    Output:
    ```json
    {{
        "inhosp_death_30day":"True",
        "explanation":{first_reason}
    }}
    ```

    **Output**:
    ----------
    `response`: dict
        - `"inhosp_death_30day"`: str, "True" (high risk) or "False" (low risk).

    **Notes**:
    ----------
    - Input data must include all required fields.
    - The output is a binary prediction of risk: `"True"` for high risk or `"False"` for low risk.
    - Examples demonstrate the expected input-output behavior.
    """

# Model Task:
Input: {patient_info}
Output:
'''.strip()

    dataset['2shot_false_prompt'] = dataset['text'].apply(lambda x: prompt_template_2shot_false.format(
        first_data=first_sentence,
        first_reason=first_reason,
        second_data=second_sentence,
        second_reason=second_reason,
        patient_info=f"'{x}'"
    ))
    return dataset