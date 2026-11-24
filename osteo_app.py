# -*- coding: utf-8 -*-

"""
骨质疏松风险预测工具（风险等级强化版）
============================================
左右分栏，无需滚动
中英文双语
ADL悬停提示
风险等级大号显示
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ==================== 加载模型 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
continuous_vars = joblib.load(os.path.join(MODEL_DIR, "continuous_vars.pkl"))

# ==================== 页面设置 ====================
st.set_page_config(
    page_title="Osteoporosis Risk Prediction",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 深度 CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans SC', -apple-system, sans-serif;
    color: #1f2937;
}

.stApp { background: #f6f8fb; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="collapsedControl"] {display: none;}
button[kind="header"] {display: none;}

.main .block-container {
    max-width: 1400px;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 56px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #eef2f6;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    z-index: 999;
}

.navbar-brand {
    font-size: 1rem;
    font-weight: 800;
    color: #0f766e;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.navbar-links {
    display: flex;
    gap: 1.5rem;
    font-size: 0.85rem;
    color: #6b7280;
}

.top-spacer { height: 64px; }

.hero-compact {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
    padding: 1rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.hero-icon { font-size: 2rem; }
.hero-text { flex: 1; }
.hero-title { font-size: 1.3rem; font-weight: 800; margin-bottom: 0.15rem; }
.hero-subtitle { font-size: 0.8rem; opacity: 0.9; }

.section-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #111827;
    margin-top: 0.6rem;
    margin-bottom: 0.3rem;
}

.section-caption {
    color: #6b7280;
    font-size: 0.78rem;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}

.adl-score-compact {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    text-align: center;
    margin-top: 0.5rem;
    color: #065f46;
    font-weight: 700;
    font-size: 0.9rem;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 2.8rem;
    font-size: 0.95rem;
    font-weight: 700;
    border: none;
    background: linear-gradient(135deg, #0f766e, #0d9488);
    color: white;
    box-shadow: 0 6px 16px -4px rgba(13, 148, 136, 0.5);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px -4px rgba(13, 148, 136, 0.6);
}

.stNumberInput input,
.stTextInput input {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    padding: 0.5rem 0.8rem !important;
    font-size: 0.9rem !important;
}

.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.12) !important;
}

.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
}

.stCheckbox { padding: 0.15rem 0; }
.stCheckbox label { font-size: 0.88rem; color: #374151; }

/* 结果卡片 */
.result-card-compact {
    background: white;
    border: 1px solid #eef2f6;
    border-radius: 18px;
    padding: 1.8rem 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.result-label {
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 0.3rem;
}

.result-number {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0.2rem 0 1rem 0;
    line-height: 1;
}

/* 风险分层标签（小） */
.result-level-label {
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

/* 风险分层（大号） */
.result-level-big {
    display: inline-block;
    padding: 0.7rem 2rem;
    border-radius: 999px;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
}

.result-level-big.low {
    color: #047857;
    background: #ecfdf5;
    border: 2px solid #a7f3d0;
}

.result-level-big.medium {
    color: #b45309;
    background: #fffbeb;
    border: 2px solid #fde68a;
}

.result-level-big.high {
    color: #b91c1c;
    background: #fef2f2;
    border: 2px solid #fecaca;
}

/* 阈值说明 */
.result-threshold {
    color: #9ca3af;
    font-size: 0.7rem;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* 建议卡片 */
.advice-card-compact {
    background: white;
    border: 1px solid #eef2f6;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-top: 0.7rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.advice-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.5rem;
}

.advice-content {
    color: #4b5563;
    line-height: 1.7;
    font-size: 0.82rem;
}

.footer-compact {
    text-align: center;
    color: #94a3b8;
    font-size: 0.7rem;
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ==================== 顶部导航栏 ====================
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">🦴 OsteoRisk</div>
    <div class="navbar-links">
        <span>Home</span>
        <span>Assessment</span>
        <span>About</span>
    </div>
</div>
<div class="top-spacer"></div>
""", unsafe_allow_html=True)


# ==================== 双语文本 ====================
TEXT = {
    "中文": {
        "title": "骨质疏松风险预测",
        "subtitle": "基于6变量简化模型，评估8年内新发骨质疏松风险",
        "basic": "👤 基本信息",
        "age": "年龄（岁）",
        "sex": "性别",
        "female": "女",
        "male": "男",
        "grip": "💪 左手握力（kg）",
        "adl": "🚶 日常生活活动能力（ADL）",
        "adl_caption": "请勾选需要他人帮助或完全不能自己完成的活动条目。总分越高，功能受损越重。",
        "bathing": "洗澡",
        "bathing_hint": "需要他人帮助才能洗澡",
        "dressing": "穿衣",
        "dressing_hint": "需要他人帮助才能穿衣",
        "eating": "吃饭",
        "eating_hint": "需要他人帮助才能吃饭",
        "transferring": "上下床",
        "transferring_hint": "需要他人帮助才能从床上移到椅子上",
        "walking": "室内行走",
        "walking_hint": "需要他人帮助或辅助器具才能在室内行走",
        "toileting": "上厕所",
        "toileting_hint": "需要他人帮助才能上厕所",
        "adl_score": "ADL总分",
        "comorbidity": "🩺 共病情况",
        "arthritis": "关节炎",
        "lung": "肺病",
        "no": "无",
        "yes": "有",
        "predict": "🔍 计算8年风险",
        "result": "📊 风险评估结果",
        "risk": "8年内新发骨质疏松的预测概率",
        "level": "风险分层（仅供参考，不是诊断）",
        "threshold_note": "低风险 <10%｜中风险 10–20%｜高风险 ≥20%",
        "low": "低风险",
        "medium": "中风险",
        "high": "高风险",
        "advice": "💡 健康管理建议",
        "low_advice": "• 保持规律负重及肌力活动\n• 保证钙和维生素D摄入\n• 定期进行骨健康评估",
        "medium_advice": "• 讨论是否需DXA骨密度检查\n• 增加负重运动和肌力训练\n• 保证钙和维生素D摄入\n• 戒烟限酒，预防跌倒",
        "high_advice": "• 尽快讨论DXA骨密度检查\n• 评估是否需要药物干预\n• 加强跌倒预防措施\n• 适当蛋白质摄入和肌力训练\n• 定期随访骨健康",
        "disclaimer": "本工具仅用于风险评估参考，不能替代专业医疗建议。"
    },
    "English": {
        "title": "Osteoporosis Risk Prediction",
        "subtitle": "6-variable model for 8-year incident osteoporosis risk",
        "basic": "👤 Basic Information",
        "age": "Age (years)",
        "sex": "Sex",
        "female": "Female",
        "male": "Male",
        "grip": "💪 Left-hand grip (kg)",
        "adl": "🚶 Activities of Daily Living (ADL)",
        "adl_caption": "Check the activities that require assistance or cannot be performed independently. Higher scores indicate greater functional impairment.",
        "bathing": "Bathing",
        "bathing_hint": "Requires assistance to bathe",
        "dressing": "Dressing",
        "dressing_hint": "Requires assistance to dress",
        "eating": "Eating",
        "eating_hint": "Requires assistance to eat",
        "transferring": "Transferring",
        "transferring_hint": "Requires assistance to move from bed to chair",
        "walking": "Walking",
        "walking_hint": "Requires assistance or aids to walk indoors",
        "toileting": "Toileting",
        "toileting_hint": "Requires assistance to use the toilet",
        "adl_score": "ADL score",
        "comorbidity": "🩺 Comorbidities",
        "arthritis": "Arthritis",
        "lung": "Lung disease",
        "no": "No",
        "yes": "Yes",
        "predict": "🔍 Calculate 8-year risk",
        "result": "📊 Risk Assessment",
        "risk": "Predicted 8-year probability of incident osteoporosis",
        "level": "Risk stratification (for reference only, not a diagnosis)",
        "threshold_note": "Low <10% | Intermediate 10–20% | High ≥20%",
        "low": "Low risk",
        "medium": "Intermediate risk",
        "high": "High risk",
        "advice": "💡 Health Management Advice",
        "low_advice": "• Regular weight-bearing exercise\n• Adequate calcium and vitamin D\n• Periodic bone health assessment",
        "medium_advice": "• Discuss DXA bone density testing\n• Increase weight-bearing exercise\n• Adequate calcium and vitamin D\n• Avoid smoking and excess alcohol",
        "high_advice": "• Discuss DXA bone density testing soon\n• Consider pharmacological intervention\n• Strengthen fall prevention\n• Adequate protein and strength training\n• Regular bone health follow-up",
        "disclaimer": "For risk assessment only. Not a substitute for professional medical advice."
    }
}

# ==================== 语言切换 ====================
lang = st.radio(
    "",
    ["中文", "English"],
    horizontal=True,
    label_visibility="collapsed"
)

t = TEXT[lang]

# ==================== 左右分栏 ====================
left_col, right_col = st.columns([1, 1], gap="large")

# ==================== 左侧：输入 ====================
with left_col:

    st.markdown(
        f"""
        <div class="hero-compact">
            <div class="hero-icon">🦴</div>
            <div class="hero-text">
                <div class="hero-title">{t["title"]}</div>
                <div class="hero-subtitle">{t["subtitle"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="section-title">{t["basic"]}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(t["age"], min_value=40, max_value=100, value=65, step=1)
    with col2:
        sex = st.selectbox(t["sex"], [t["female"], t["male"]])
    sex_code = 0 if sex == t["female"] else 1

    lgrip = st.number_input(t["grip"], min_value=0.0, max_value=80.0, value=25.0, step=0.5)

    st.markdown(f'<div class="section-title">{t["adl"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">{t["adl_caption"]}</div>', unsafe_allow_html=True)

    adl_col1, adl_col2 = st.columns(2)
    adl_values = {}

    with adl_col1:
        adl_values["bathing"] = st.checkbox(
            t["bathing"], value=False, help=t["bathing_hint"]
        )
        adl_values["dressing"] = st.checkbox(
            t["dressing"], value=False, help=t["dressing_hint"]
        )
        adl_values["eating"] = st.checkbox(
            t["eating"], value=False, help=t["eating_hint"]
        )

    with adl_col2:
        adl_values["transferring"] = st.checkbox(
            t["transferring"], value=False, help=t["transferring_hint"]
        )
        adl_values["walking"] = st.checkbox(
            t["walking"], value=False, help=t["walking_hint"]
        )
        adl_values["toileting"] = st.checkbox(
            t["toileting"], value=False, help=t["toileting_hint"]
        )

    adl_total = sum(1 for v in adl_values.values() if v)

    st.markdown(
        f'<div class="adl-score-compact">{t["adl_score"]}：{adl_total} / 6</div>',
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="section-title">{t["comorbidity"]}</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        arthritis = st.selectbox(t["arthritis"], [t["no"], t["yes"]])
    arthritis_code = 0 if arthritis == t["no"] else 1
    with col4:
        lung_disease = st.selectbox(t["lung"], [t["no"], t["yes"]])
    lung_disease_code = 0 if lung_disease == t["no"] else 1

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button(t["predict"], type="primary")


# ==================== 右侧：结果 ====================
with right_col:

    if predict_clicked:

        input_data = pd.DataFrame({
            "lgrip": [lgrip],
            "sex": [sex_code],
            "arthritis": [arthritis_code],
            "age": [age],
            "adl_total": [adl_total],
            "lung_disease": [lung_disease_code],
        })

        input_data = input_data[features]

        input_imp = imputer.transform(input_data)
        input_imp = pd.DataFrame(input_imp, columns=features)

        for v in features:
            if v not in continuous_vars:
                input_imp[v] = np.round(input_imp[v])

        input_scaled = scaler.transform(input_imp)
        prob = model.predict_proba(input_scaled)[0, 1]

        if prob < 0.10:
            risk_level = t["low"]
            risk_class = "low"
            risk_icon = "✅"
        elif prob < 0.20:
            risk_level = t["medium"]
            risk_class = "medium"
            risk_icon = "⚠️"
        else:
            risk_level = t["high"]
            risk_class = "high"
            risk_icon = "🚨"

        st.markdown(f'<div class="section-title">{t["result"]}</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="result-card-compact">
                <div class="result-label">{t["risk"]}</div>
                <div class="result-number {risk_class}">{prob * 100:.1f}%</div>
                <div class="result-level-label">{t["level"]}</div>
                <div class="result-level-big {risk_class}">{risk_icon} {risk_level}</div>
                <div class="result-threshold">{t["threshold_note"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(min(float(prob), 1.0))

        if risk_class == "low":
            advice = t["low_advice"]
        elif risk_class == "medium":
            advice = t["medium_advice"]
        else:
            advice = t["high_advice"]

        st.markdown(
            f"""
            <div class="advice-card-compact">
                <div class="advice-title">{t["advice"]}</div>
                <div class="advice-content">{advice.replace(chr(10), "<br>")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div style="height: 100%; display: flex; align-items: center; justify-content: center; 
                        min-height: 500px; color: #cbd5e1; font-size: 0.95rem;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                    <div>{t["result"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==================== Footer ====================
st.markdown(
    f"""
    <div class="footer-compact">
        {t["disclaimer"]}
    </div>
    """,
    unsafe_allow_html=True
)
