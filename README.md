# Osteoporosis Risk Prediction App

Streamlit web application for predicting **8-year incident osteoporosis risk** using a 6-variable logistic regression model.

基于 6 变量逻辑回归模型的骨质疏松 8 年新发风险预测网页工具，支持中英文双语界面。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3x-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- 6-variable simplified model: age, sex, left-hand grip strength, ADL total score, arthritis, lung disease
- Bilingual interface (Chinese / English)
- Instant risk probability with three-level stratification: low <10%, intermediate 10-20%, high >=20%
- Built-in health management advice per risk level
- Compact two-column layout, ready for screenshots in manuscripts

## Model pipeline

```
raw input -> column reorder (features.pkl) -> imputation (imputer.pkl)
          -> rounding of categorical vars -> standardization (scaler.pkl)
          -> logistic regression -> predicted probability
```

## Project structure

```
.
├── osteo_app.py              # Streamlit entry script
├── model/                    # serialized model artifacts
│   ├── logistic_model.pkl
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── continuous_vars.pkl
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run osteo_app.py
```

Then open http://localhost:8501

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, choose this repository, set main file path to `osteo_app.py`.
4. Deploy. Model files under `model/` are loaded through relative paths, so no extra configuration is needed.

## Reproducibility notes

- Feature order in `features.pkl` must match the order used during model training.
- Categorical encoding: sex (0 = female, 1 = male); arthritis and lung_disease (0 = no, 1 = yes); adl_total = number of assisted ADL items (0-6).
- Preprocessing steps must be applied in the exact order above; do not reorder.

## Disclaimer

This tool is intended for research and risk-assessment reference only. It does not constitute a diagnosis and must not replace professional medical advice.

## License

MIT

> （注：部分内容由豆包 AI 生成）