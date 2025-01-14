# LLM-vs-ML-PostOp

This repository contains the code and resources for the study **"Large Language Models vs. Machine Learning for Predicting Postoperative Outcomes"**. The study investigates the potential of large language models (LLMs) as scalable alternatives to traditional machine learning models (e.g., XGBoost) for predicting critical postoperative outcomes, such as in-hospital 30-day mortality, ICU admission, and acute kidney injury.

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
