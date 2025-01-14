# LLM-vs-ML-PostOp

This repository contains the code and resources for the study **"Large Language Models vs. Machine Learning for Predicting Postoperative Outcomes"**. The study investigates the potential of large language models (LLMs) as scalable alternatives to traditional machine learning models (e.g., XGBoost) for predicting critical postoperative outcomes, such as in-hospital 30-day mortality, ICU admission, and acute kidney injury.

---

## Abstract

Artificial intelligence has revolutionized medicine, yet the development of generalizable models remains constrained by data scarcity, heterogeneity, and stringent privacy regulations. This study investigates the potential of large language models (LLMs)—GPT-4o, Llama-3-70B, and OpenBioLLM-70B—as scalable alternatives to conventional machine learning models (XGBoost) for predicting critical postoperative outcomes, including in-hospital 30-day mortality, ICU admission, and acute kidney injury. Using real-world perioperative datasets from South Korea (INSPIRE) and the United States (MOVER), we evaluated model performance through internal and external validation. Results demonstrated that LLMs, particularly OpenBioLLM-70B, achieved comparable discriminative performance to XGBoost in predicting 30-day mortality and acute kidney injury, while outperforming XGBoost in external validation for ICU admission prediction (p < 0.05). Open-source LLMs (OpenBioLLM-70B and Llama-3-70B) also maintained performance with 4-bit quantization, reducing computational requirements by 75%, and demonstrated compatibility with on-premises deployment, addressing critical data privacy concerns. These findings highlight the potential of LLMs as versatile, efficient tools for clinical decision support, though further validation across diverse datasets and improved calibration techniques are needed to enhance their reliability and applicability in real-world healthcare settings.

---

## Repository Structure

```
LLM-vs-ML-PostOp/
├── config/               # Configuration files (e.g., environment variables)
├── data/                 # Data loading and preprocessing scripts
├── models/               # Model setup and training scripts
├── utils/                # Utility functions (e.g., text conversion, log probability extraction)
├── main.py               # Main script to run the pipeline
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jipyeong-lee/LLM-vs-ML-PostOp.git
   cd LLM-vs-ML-PostOp
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory and add the following:
     ```plaintext
     DATA_PATH=/path/to/your/data
     MODEL_PATH=/path/to/your/model
     OUTPUT_PATH=/path/to/your/output
     ```

---

## Usage

To run the pipeline, execute the following command:

```bash
python main.py
```

This will:
1. Load and preprocess the data.
2. Train and evaluate the models (LLMs and XGBoost).
3. Generate predictions and save the results.

---

## Key Findings

- **Comparable Performance**: LLMs (especially OpenBioLLM-70B) achieved comparable performance to XGBoost in predicting 30-day mortality and acute kidney injury.
- **External Validation**: LLMs outperformed XGBoost in external validation for ICU admission prediction (p < 0.05).
- **Efficiency**: Open-source LLMs maintained performance with 4-bit quantization, reducing computational requirements by 75%.
- **Privacy**: LLMs demonstrated compatibility with on-premises deployment, addressing data privacy concerns.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or findings in your research, please cite our work:

```plaintext
[To be updated with the full citation once the paper is published.]
```

---

## Contact

For questions or feedback, please contact [Jipyeong Lee] at [easypyeong@gmail.com].
```
