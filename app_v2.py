import os
import time
import uuid
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# FRAUDGUARD
# FRAUD INTELLIGENCE PLATFORM
# ============================================================

st.set_page_config(
    page_title="FraudGuard | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "credit_card_fraud_dataset.csv",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fraud_detection_model.pkl",
)

MERCHANT_FREQ_PATH = os.path.join(
    BASE_DIR,
    "merchant_frequency.pkl",
)


# ============================================================
# MODEL CONTRACT
# KEEP THIS EXACTLY ALIGNED WITH THE EXISTING MODEL
# ============================================================

MODEL_FEATURES = [
    "Amount",
    "TransactionMonth",
    "TransactionDay",
    "TransactionDayOfWeek",
    "TransactionType_refund",
    "Location_Dallas",
    "Location_Houston",
    "Location_Los Angeles",
    "Location_New York",
    "Location_Philadelphia",
    "Location_Phoenix",
    "Location_San Antonio",
    "Location_San Diego",
    "Location_San Jose",
    "MerchantFrequency",
]


LOCATIONS = [
    "Chicago",
    "Dallas",
    "Houston",
    "Los Angeles",
    "New York",
    "Philadelphia",
    "Phoenix",
    "San Antonio",
    "San Diego",
    "San Jose",
]


TRANSACTION_TYPES = [
    "purchase",
    "refund",
]


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap'
);


/* =========================================================
   VARIABLES
   ========================================================= */

:root {
    --bg: #060714;
    --bg2: #0A0C1B;
    --panel: #101329;
    --panel2: #141832;

    --cyan: #39E7FF;
    --cyan2: #00C8FF;

    --violet: #8A5CFF;
    --violet2: #5D38FF;

    --pink: #FF4EDB;
    --lime: #B7FF63;
    --yellow: #FFD166;
    --red: #FF5577;

    --white: #F7F8FF;
    --muted: #858DAE;
    --muted2: #626A87;

    --line: rgba(255,255,255,0.075);
}


/* =========================================================
   APPLICATION
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(138,92,255,0.16),
            transparent 27%
        ),
        radial-gradient(
            circle at 5% 85%,
            rgba(57,231,255,0.07),
            transparent 30%
        ),
        radial-gradient(
            circle at 100% 80%,
            rgba(255,78,219,0.06),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #050611 0%,
            #080A19 50%,
            #10091D 100%
        );

    color: var(--white);
}


.main .block-container {
    max-width: 1500px;
    padding-top: 1.7rem;
    padding-bottom: 4rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #080A18 0%,
            #0B0F23 55%,
            #110D24 100%
        );

    border-right:
        1px solid rgba(57,231,255,0.10);

    min-width: 292px;
    max-width: 292px;
}


[data-testid="stSidebar"] > div:first-child {
    padding: 0 16px 18px 16px;
}


[data-testid="stSidebar"] section {
    padding-top: 0 !important;
}


/* =========================================================
   SIDEBAR BRAND
   ========================================================= */

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;

    padding: 4px 5px 18px 5px;

    animation:
        fadeSlide 0.55s ease both;
}


.sidebar-logo {
    width: 48px;
    height: 48px;

    flex: 0 0 48px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 14px;

    background:
        linear-gradient(
            145deg,
            rgba(57,231,255,0.14),
            rgba(138,92,255,0.20)
        );

    border:
        1px solid rgba(57,231,255,0.22);

    box-shadow:
        0 0 25px rgba(57,231,255,0.08),
        inset 0 1px rgba(255,255,255,0.10);

    font-size: 24px;

    animation:
        logoPulse 3s ease-in-out infinite;
}


.sidebar-brand-text {
    min-width: 0;
}


.sidebar-title {
    color: #FFFFFF;

    font-size: 22px;
    font-weight: 900;

    letter-spacing: -0.04em;
}


.sidebar-subtitle {
    color: #68718E;

    font-family:
        'Orbitron',
        sans-serif;

    font-size: 7px;
    font-weight: 700;

    letter-spacing: 0.09em;

    margin-top: 5px;

    white-space: nowrap;
}


/* =========================================================
   WORKSPACE
   ========================================================= */

.sidebar-label {
    color: #22D3EE;

    font-family:
        'Orbitron',
        sans-serif;

    font-size: 11px;
    font-weight: 800;

    letter-spacing: 0.16em;

    padding: 8px 9px 10px 9px;
}


/* =========================================================
   NAVIGATION
   ========================================================= */

[data-testid="stSidebar"] .stButton {
    width: 100%;
    margin: 0 0 4px 0;
}


[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;

    min-height: 37px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    padding: 0 12px !important;

    border-radius: 9px !important;

    border:
        1px solid rgba(255,255,255,0.045) !important;

    background:
        rgba(255,255,255,0.016) !important;

    color:
        #9199B7 !important;

    box-shadow:
        none !important;

    font-size:
        10px !important;

    font-weight:
        800 !important;

    letter-spacing:
        0.055em !important;

    transition:
        all 0.18s ease !important;
}


[data-testid="stSidebar"] .stButton > button:hover {
    background:
        linear-gradient(
            90deg,
            rgba(57,231,255,0.07),
            rgba(138,92,255,0.07)
        ) !important;

    border-color:
        rgba(57,231,255,0.15) !important;

    color:
        #FFFFFF !important;

    transform:
        translateX(3px) !important;
}


[data-testid="stSidebar"]
.stButton > button[kind="primary"] {

    background:
        linear-gradient(
            90deg,
            rgba(57,231,255,0.12),
            rgba(138,92,255,0.14)
        ) !important;

    border:
        1px solid rgba(57,231,255,0.25) !important;

    color:
        #FFFFFF !important;

    box-shadow:
        inset 3px 0 0 #39E7FF,
        0 8px 25px rgba(57,231,255,0.05) !important;
}


/* =========================================================
   SIDEBAR ENGINE STATUS
   ========================================================= */

.sidebar-status {
    margin: 22px 3px 0 3px;

    padding: 14px;

    border-radius: 14px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.04),
            rgba(255,255,255,0.012)
        );

    border:
        1px solid rgba(255,255,255,0.065);
}


.sidebar-status-title {
    display: flex;
    align-items: center;
    gap: 8px;

    color: #E9EBF8;

    font-size: 9px;
    font-weight: 900;

    letter-spacing: 0.10em;
}


.status-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: var(--lime);

    box-shadow:
        0 0 12px rgba(183,255,99,0.85);

    animation:
        statusPulse 1.8s ease-in-out infinite;
}


.sidebar-status-text {
    color: #707896;

    font-size: 9px;
    line-height: 1.5;

    margin-top: 8px;
}


.sidebar-status-model {
    display: flex;
    justify-content: space-between;

    margin-top: 11px;
    padding-top: 10px;

    border-top:
        1px solid rgba(255,255,255,0.055);

    color: #646C89;

    font-size: 8px;
}


.sidebar-status-model strong {
    color: #E7E9F6;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    position: relative;
    overflow: hidden;

    padding: 32px;

    border-radius: 25px;

    border:
        1px solid rgba(57,231,255,0.14);

    background:
        radial-gradient(
            circle at 92% 15%,
            rgba(138,92,255,0.25),
            transparent 28%
        ),
        radial-gradient(
            circle at 8% 100%,
            rgba(57,231,255,0.09),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            rgba(16,27,55,0.96),
            rgba(25,12,45,0.96)
        );

    box-shadow:
        0 22px 65px rgba(0,0,0,0.24),
        inset 0 1px rgba(255,255,255,0.05);

    animation:
        fadeSlide 0.55s ease both;
}


.hero::after {
    content: "";

    position: absolute;

    width: 45%;
    height: 1px;

    left: -50%;

    top: 0;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(57,231,255,0.8),
            transparent
        );

    animation:
        scanSweep 4.5s ease-in-out infinite;
}


.hero-eyebrow {
    color: var(--cyan);

    font-family:
        'Orbitron',
        sans-serif;

    font-size: 8px;
    font-weight: 800;

    letter-spacing: 0.19em;
}


.hero-title {
    color: #FFFFFF;

    font-size: 50px;
    font-weight: 950;

    letter-spacing: -0.055em;

    line-height: 1;

    margin-top: 12px;
}


.hero-title span {
    background:
        linear-gradient(
            90deg,
            #FFFFFF,
            #BDF7FF,
            #D7C4FF
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.hero-description {
    max-width: 920px;

    color: #AEB7D5;

    font-size: 13px;
    line-height: 1.65;

    margin-top: 13px;
}


.hero-status {
    display: inline-flex;

    align-items: center;
    gap: 8px;

    margin-top: 18px;

    padding: 7px 11px;

    border-radius: 999px;

    background:
        rgba(183,255,99,0.05);

    border:
        1px solid rgba(183,255,99,0.17);

    color: var(--lime);

    font-size: 8px;

    font-weight: 900;

    letter-spacing: 0.10em;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-header {
    display: flex;

    justify-content: space-between;
    align-items: end;

    margin: 27px 0 12px 0;

    animation:
        fadeSlide 0.45s ease both;
}


.section-header-title {
    color: #F5F6FF;

    font-size: 16px;
    font-weight: 900;
}


.section-header-subtitle {
    color: #68718F;

    font-size: 9px;

    text-align: right;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    min-height: 112px;

    padding: 17px;

    border-radius: 17px;

    background:
        linear-gradient(
            145deg,
            rgba(17,21,47,0.94),
            rgba(10,13,30,0.94)
        );

    border:
        1px solid rgba(255,255,255,0.07);

    box-shadow:
        0 14px 40px rgba(0,0,0,0.15);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;

    animation:
        cardAppear 0.55s ease both;
}


.metric-card:hover {
    transform: translateY(-4px);

    border-color:
        rgba(57,231,255,0.18);
}


.metric-label {
    color: #9AA4C2;

    font-size: 11px;
    font-weight: 900;

    letter-spacing: 0.10em;
}


.metric-value {
    color: #FFFFFF;

    font-size: 26px;
    font-weight: 950;

    letter-spacing: -0.045em;

    margin-top: 7px;
}


.metric-hint {
    color: #AEB7D5;
    font-size: 11px;
    font-weight: 500;
    margin-top: 8px;
}


.accent-cyan .metric-value {
    color: var(--cyan);
}

.accent-violet .metric-value {
    color: #B28EFF;
}

.accent-pink .metric-value {
    color: var(--pink);
}

.accent-lime .metric-value {
    color: var(--lime);
}

.accent-yellow .metric-value {
    color: var(--yellow);
}


/* =========================================================
   PANELS
   ========================================================= */

.info-card {
    padding: 20px;

    border-radius: 17px;

    background:
        linear-gradient(
            145deg,
            rgba(17,21,47,0.94),
            rgba(10,13,30,0.94)
        );

    border:
        1px solid rgba(255,255,255,0.07);
}


.panel-title {
    color: #F3F5FF;

    font-size: 13px;
    font-weight: 900;

    letter-spacing: 0.02em;
}


.panel-subtitle {
    color: #69718E;

    font-size: 8px;

    margin-top: 4px;
}


/* =========================================================
   RISK BADGES
   ========================================================= */

.risk-badge {
    display: inline-flex;

    align-items: center;
    justify-content: center;

    padding: 5px 9px;

    border-radius: 999px;

    font-size: 8px;
    font-weight: 900;

    letter-spacing: 0.07em;
}


.risk-low {
    color: var(--lime);

    background:
        rgba(183,255,99,0.08);

    border:
        1px solid rgba(183,255,99,0.18);
}


.risk-medium {
    color: var(--yellow);

    background:
        rgba(255,209,102,0.08);

    border:
        1px solid rgba(255,209,102,0.18);
}


.risk-high {
    color: #FF9C66;

    background:
        rgba(255,120,70,0.09);

    border:
        1px solid rgba(255,120,70,0.20);
}


.risk-critical {
    color: var(--pink);

    background:
        rgba(255,78,219,0.09);

    border:
        1px solid rgba(255,78,219,0.22);

    animation:
        criticalPulse 1.6s ease-in-out infinite;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    min-height: 43px !important;

    border-radius: 11px !important;

    border:
        1px solid rgba(255,255,255,0.08) !important;

    background:
        linear-gradient(
            90deg,
            #5D38FF,
            #D342FF
        ) !important;

    color: #FFFFFF !important;

    font-weight: 900 !important;

    letter-spacing: 0.03em !important;

    box-shadow:
        0 12px 32px rgba(122,61,255,0.20) !important;

    transition:
        all 0.2s ease !important;
}


.stButton > button:hover {
    transform:
        translateY(-2px) !important;

    filter:
        brightness(1.07) !important;

    box-shadow:
        0 16px 40px rgba(122,61,255,0.30) !important;
}


/* =========================================================
   INPUTS
   ========================================================= */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
[data-testid="stDateInput"] input {

    background:
        #11152A !important;

    color:
        #FFFFFF !important;

    border:
        1px solid rgba(255,255,255,0.10) !important;

    border-radius:
        10px !important;
}


label {
    color:
        #AEB5D0 !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background:
        rgba(255,255,255,0.025);

    border:
        1px dashed rgba(57,231,255,0.25);

    border-radius:
        14px;

    padding:
        10px;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    border-radius:
        13px;

    overflow:
        hidden;

    border:
        1px solid rgba(255,255,255,0.07);
}


/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"] {
    background:
        rgba(14,17,37,0.75);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:
        13px;
}


/* =========================================================
   PROGRESS
   ========================================================= */

.stProgress > div > div > div > div {
    background:
        linear-gradient(
            90deg,
            #39E7FF,
            #8A5CFF,
            #FF4EDB
        ) !important;
}


/* =========================================================
   ANIMATION
   ========================================================= */

@keyframes fadeSlide {
    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


@keyframes cardAppear {
    from {
        opacity: 0;
        transform: translateY(10px) scale(0.985);
    }

    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}


@keyframes logoPulse {
    0%, 100% {
        box-shadow:
            0 0 22px rgba(57,231,255,0.07),
            inset 0 1px rgba(255,255,255,0.10);
    }

    50% {
        box-shadow:
            0 0 32px rgba(57,231,255,0.16),
            inset 0 1px rgba(255,255,255,0.12);
    }
}


@keyframes statusPulse {
    0%, 100% {
        opacity: 0.55;
        transform: scale(0.9);
    }

    50% {
        opacity: 1;
        transform: scale(1.2);
    }
}


@keyframes criticalPulse {
    0%, 100% {
        box-shadow:
            0 0 0 rgba(255,78,219,0);
    }

    50% {
        box-shadow:
            0 0 18px rgba(255,78,219,0.18);
    }
}


@keyframes scanSweep {
    0% {
        left: -50%;
        opacity: 0;
    }

    15% {
        opacity: 1;
    }

    70% {
        opacity: 0.9;
    }

    100% {
        left: 110%;
        opacity: 0;
    }
}

/* =========================================================
   FRAUDGUARD MOTION 2.0
   PREMIUM SOC DASHBOARD MOTION SYSTEM
   ========================================================= */


/* =========================================================
   PAGE ENTRANCE
   ========================================================= */

.main .block-container {
    animation:
        fgPageIn
        0.55s
        cubic-bezier(0.22, 1, 0.36, 1)
        both;
}


@keyframes fgPageIn {

    0% {
        opacity: 0;
        transform:
            translateY(10px);
    }

    100% {
        opacity: 1;
        transform:
            translateY(0);
    }

}


/* =========================================================
   HERO MOTION
   ========================================================= */

.hero-title {
    animation:
        fgHeroIn
        0.60s
        cubic-bezier(0.22, 1, 0.36, 1)
        both;
}


.hero-subtitle {
    animation:
        fgHeroIn
        0.75s
        cubic-bezier(0.22, 1, 0.36, 1)
        both;
}


@keyframes fgHeroIn {

    0% {
        opacity: 0;
        transform:
            translateY(12px);
    }

    100% {
        opacity: 1;
        transform:
            translateY(0);
    }

}


/* =========================================================
   HERO GLOW
   ========================================================= */

.hero,
.hero-title {
    position:
        relative;
}


.hero::before {
    content:
        "";

    position:
        absolute;

    inset:
        -20px;

    pointer-events:
        none;

    border-radius:
        24px;

    background:
        radial-gradient(
            circle at 20% 50%,
            rgba(57,231,255,0.10),
            transparent 35%
        );

    opacity:
        0;

    animation:
        fgHeroGlow
        6s
        ease-in-out
        infinite;
}


@keyframes fgHeroGlow {

    0%,
    100% {
        opacity:
            0.25;
    }

    50% {
        opacity:
            0.75;
    }

}


/* =========================================================
   HERO SCAN BEAM
   ========================================================= */

.hero::after {
    content:
        "";

    position:
        absolute;

    top:
        0;

    bottom:
        0;

    left:
        -60%;

    width:
        35%;

    pointer-events:
        none;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(57,231,255,0.08),
            rgba(57,231,255,0.18),
            rgba(57,231,255,0.08),
            transparent
        );

    transform:
        skewX(-18deg);

    animation:
        fgHeroScan
        7s
        ease-in-out
        infinite;
}


@keyframes fgHeroScan {

    0% {
        left:
            -60%;
        opacity:
            0;
    }

    15% {
        opacity:
            1;
    }

    70% {
        opacity:
            0.8;
    }

    100% {
        left:
            120%;
        opacity:
            0;
    }

}


/* =========================================================
   PREMIUM LOGO MOTION
   ========================================================= */

.sidebar-logo {
    animation:
        premiumLogoGlow
        4.5s
        ease-in-out
        infinite;
}


@keyframes premiumLogoGlow {

    0%,
    100% {
        box-shadow:
            0 0 20px rgba(57,231,255,0.07),
            inset 0 1px rgba(255,255,255,0.10);

        border-color:
            rgba(57,231,255,0.20);
    }

    50% {
        box-shadow:
            0 0 27px rgba(57,231,255,0.13),
            0 0 42px rgba(138,92,255,0.05),
            inset 0 1px rgba(255,255,255,0.12);

        border-color:
            rgba(57,231,255,0.27);
    }

}


/* =========================================================
   SIDEBAR BRAND
   ========================================================= */

.sidebar-brand {
    animation:
        fadeSlide
        0.55s
        ease
        both;
}


.sidebar-title {
    animation:
        fgTextReveal
        0.65s
        ease
        both;
}


.sidebar-subtitle {
    animation:
        fgTextReveal
        0.85s
        ease
        both;
}


@keyframes fgTextReveal {

    0% {
        opacity:
            0;
        transform:
            translateX(-6px);
    }

    100% {
        opacity:
            1;
        transform:
            translateX(0);
    }

}


/* =========================================================
   STATUS INDICATORS
   ========================================================= */

.status-dot {
    animation:
        statusPulse
        2.2s
        ease-in-out
        infinite;
}


/* Generic live indicator */

.live-indicator,
.live-dot,
.system-status-dot {
    animation:
        statusPulse
        2s
        ease-in-out
        infinite;
}


/* =========================================================
   KPI / METRIC CARDS
   ========================================================= */

.metric-card,
.kpi-card,
.custom-card,
.prediction-card,
.system-card,
.info-card {
    animation:
        cardAppear
        0.55s
        cubic-bezier(0.22, 1, 0.36, 1)
        both;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}


.metric-card:hover,
.kpi-card:hover,
.custom-card:hover,
.prediction-card:hover,
.system-card:hover,
.info-card:hover {

    transform:
        translateY(-5px);

    border-color:
        rgba(57,231,255,0.20);

    box-shadow:
        0 12px 32px rgba(0,0,0,0.30),
        0 0 24px rgba(57,231,255,0.07);
}


/* =========================================================
   KPI NUMBER EMPHASIS
   ========================================================= */

.metric-card [data-testid="stMetricValue"],
.kpi-card [data-testid="stMetricValue"] {

    transition:
        transform 0.25s ease,
        text-shadow 0.25s ease;
}


.metric-card:hover [data-testid="stMetricValue"],
.kpi-card:hover [data-testid="stMetricValue"] {

    transform:
        scale(1.025);

    text-shadow:
        0 0 18px rgba(57,231,255,0.18);
}


/* =========================================================
   BUTTON INTERACTION
   ========================================================= */

.stButton > button {

    position:
        relative;

    overflow:
        hidden;

    transition:
        transform 0.20s ease,
        box-shadow 0.20s ease,
        border-color 0.20s ease;
}


.stButton > button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 8px 24px rgba(57,231,255,0.12);
}


.stButton > button::after {

    content:
        "";

    position:
        absolute;

    top:
        0;

    left:
        -120%;

    width:
        60%;

    height:
        100%;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.10),
            transparent
        );

    transform:
        skewX(-20deg);

    transition:
        left 0.55s ease;

    pointer-events:
        none;
}


.stButton > button:hover::after {

    left:
        140%;
}


/* =========================================================
   SIDEBAR NAVIGATION
   ========================================================= */

[data-testid="stSidebar"] button {

    transition:
        transform 0.20s ease,
        background 0.20s ease,
        box-shadow 0.20s ease,
        border-color 0.20s ease;
}


[data-testid="stSidebar"] button:hover {

    transform:
        translateX(3px);

    box-shadow:
        0 0 18px rgba(57,231,255,0.07);
}


/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-title,
.section-header,
.nav-label {

    animation:
        fadeSlide
        0.45s
        ease
        both;
}


/* =========================================================
   RISK SCORE
   ========================================================= */

.risk-score,
.risk-score-card,
.risk-score-container {

    animation:
        fgRiskAppear
        0.70s
        cubic-bezier(0.22, 1, 0.36, 1)
        both;
}


@keyframes fgRiskAppear {

    0% {
        opacity:
            0;

        transform:
            scale(0.94);
    }

    100% {
        opacity:
            1;

        transform:
            scale(1);
    }

}


/* Risk score pulse when high/critical */

.risk-high,
.risk-critical,
.high-risk,
.critical-risk {

    animation:
        criticalPulse
        2.2s
        ease-in-out
        infinite;
}


/* =========================================================
   RISK PROGRESS BAR
   ========================================================= */

.stProgress > div > div > div > div {

    background-size:
        200% 100% !important;

    animation:
        fgProgressFlow
        3s
        linear
        infinite;
}


@keyframes fgProgressFlow {

    0% {
        background-position:
            100% 0;
    }

    100% {
        background-position:
            -100% 0;
    }

}


/* =========================================================
   LIVE MONITOR
   ========================================================= */

.live-monitor,
.live-feed,
.monitor-panel {

    position:
        relative;

    overflow:
        hidden;
}


/* Scanning beam hook */

.live-monitor::after,
.live-feed::after,
.monitor-panel::after {

    content:
        "";

    position:
        absolute;

    top:
        0;

    bottom:
        0;

    left:
        -45%;

    width:
        20%;

    pointer-events:
        none;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(57,231,255,0.08),
            rgba(57,231,255,0.22),
            rgba(57,231,255,0.08),
            transparent
        );

    transform:
        skewX(-15deg);

    animation:
        fgMonitorScan
        5s
        ease-in-out
        infinite;
}


@keyframes fgMonitorScan {

    0% {
        left:
            -45%;

        opacity:
            0;
    }

    15% {
        opacity:
            1;
    }

    75% {
        opacity:
            0.75;
    }

    100% {
        left:
            125%;

        opacity:
            0;
    }

}


/* =========================================================
   LIVE TRANSACTION ROWS
   ========================================================= */

.transaction-row,
.live-row,
.feed-row {

    animation:
        fgRowIn
        0.45s
        ease-out
        both;

    transition:
        background 0.20s ease,
        transform 0.20s ease;
}


.transaction-row:hover,
.live-row:hover,
.feed-row:hover {

    transform:
        translateX(3px);
}


/* =========================================================
   ALERT / CRITICAL TRANSACTION
   ========================================================= */

.alert-pulse,
.fraud-alert,
.critical-alert {

    animation:
        fgAlertPulse
        1.8s
        ease-in-out
        infinite;
}


@keyframes fgAlertPulse {

    0%,
    100% {
        box-shadow:
            0 0 0
            rgba(255,78,219,0);
    }

    50% {
        box-shadow:
            0 0 24px
            rgba(255,78,219,0.16);
    }

}


/* =========================================================
   INVESTIGATION WORKSPACE
   ========================================================= */

.investigation-panel,
.case-panel,
.analyst-panel {

    animation:
        fgPanelIn
        0.55s
        cubic-bezier(0.22, 1, 0.36, 1)
        both;
}


@keyframes fgPanelIn {

    0% {
        opacity:
            0;

        transform:
            translateX(10px);
    }

    100% {
        opacity:
            1;

        transform:
            translateX(0);
    }

}


/* =========================================================
   BULK SCANNER PIPELINE
   ========================================================= */

.pipeline,
.scanner-pipeline {

    animation:
        fadeSlide
        0.55s
        ease
        both;
}


.pipeline-stage,
.scanner-stage {

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}


.pipeline-stage:hover,
.scanner-stage:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 8px 24px rgba(57,231,255,0.08);
}


/* Active scanning stage */

.pipeline-stage.active,
.scanner-stage.active,
.stage-active {

    animation:
        fgStagePulse
        1.8s
        ease-in-out
        infinite;
}


@keyframes fgStagePulse {

    0%,
    100% {
        box-shadow:
            0 0 0
            rgba(57,231,255,0);
    }

    50% {
        box-shadow:
            0 0 20px
            rgba(57,231,255,0.14);
    }

}


/* =========================================================
   INTELLIGENCE WORKSPACE
   ========================================================= */

.intelligence-card,
.analytics-card,
.observation-card {

    animation:
        cardAppear
        0.55s
        ease
        both;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}


.intelligence-card:hover,
.analytics-card:hover,
.observation-card:hover {

    transform:
        translateY(-4px);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.25),
        0 0 18px rgba(138,92,255,0.08);
}


/* =========================================================
   CHART CONTAINERS
   ========================================================= */

.chart-container,
.analytics-chart {

    animation:
        fgChartIn
        0.65s
        ease-out
        both;
}


@keyframes fgChartIn {

    0% {
        opacity:
            0;

        transform:
            translateY(8px)
            scale(0.99);
    }

    100% {
        opacity:
            1;

        transform:
            translateY(0)
            scale(1);
    }

}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {

    transition:
        box-shadow 0.25s ease,
        border-color 0.25s ease;
}


[data-testid="stDataFrame"]:hover {

    border-color:
        rgba(57,231,255,0.16);

    box-shadow:
        0 8px 28px rgba(0,0,0,0.20);
}


/* =========================================================
   EXPANDERS
   ========================================================= */

[data-testid="stExpander"] {

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}


[data-testid="stExpander"]:hover {

    transform:
        translateY(-2px);

    border-color:
        rgba(138,92,255,0.20);

    box-shadow:
        0 8px 25px rgba(0,0,0,0.18);
}


/* =========================================================
   AMBIENT SOC GRID
   ========================================================= */

.stApp::before {

    content:
        "";

    position:
        fixed;

    inset:
        0;

    pointer-events:
        none;

    z-index:
        0;

    opacity:
        0.035;

    background-image:
        linear-gradient(
            rgba(57,231,255,0.35) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(57,231,255,0.35) 1px,
            transparent 1px
        );

    background-size:
        42px 42px;

    mask-image:
        linear-gradient(
            to bottom,
            black,
            transparent 85%
        );
}


/* =========================================================
   SUBTLE SHIMMER
   ========================================================= */

.shimmer,
.loading-shimmer {

    background:
        linear-gradient(
            110deg,
            transparent 20%,
            rgba(255,255,255,0.05) 45%,
            transparent 70%
        );

    background-size:
        250% 100%;

    animation:
        fgShimmer
        3.5s
        linear
        infinite;
}


@keyframes fgShimmer {

    0% {
        background-position:
            200% 0;
    }

    100% {
        background-position:
            -50% 0;
    }

}


/* =========================================================
   ROW ENTRANCE
   ========================================================= */

@keyframes fgRowIn {

    0% {
        opacity:
            0;

        transform:
            translateY(5px);
    }

    100% {
        opacity:
            1;

        transform:
            translateY(0);
    }

}


/* =========================================================
   STAGGERED CARD ENTRANCE
   ========================================================= */

.metric-card:nth-child(1),
.kpi-card:nth-child(1) {
    animation-delay:
        0.05s;
}


.metric-card:nth-child(2),
.kpi-card:nth-child(2) {
    animation-delay:
        0.10s;
}


.metric-card:nth-child(3),
.kpi-card:nth-child(3) {
    animation-delay:
        0.15s;
}


.metric-card:nth-child(4),
.kpi-card:nth-child(4) {
    animation-delay:
        0.20s;
}


/* =========================================================
   MOTION SAFETY
   ========================================================= */

@media (prefers-reduced-motion: reduce) {

    .hero::before,
    .hero::after,
    .live-monitor::after,
    .live-feed::after,
    .monitor-panel::after,
    .stApp::before {
        animation:
            none !important;
    }

}

/* =========================================================
   BULK SCANNER PIPELINE FLOW
   ========================================================= */

.bulk-pipeline-card {
    position: relative;
    transition:
        transform 0.28s ease,
        box-shadow 0.28s ease,
        border-color 0.28s ease;
}


/* Card hover */

.bulk-pipeline-card:hover {
    transform:
        translateY(-4px);

    border-color:
        rgba(57,231,255,0.28);

    box-shadow:
        0 14px 32px rgba(0,0,0,0.28),
        0 0 24px rgba(57,231,255,0.07);
}


/* Subtle active glow */

.bulk-pipeline-card {
    animation:
        pipelineCardIn
        0.55s
        cubic-bezier(0.22, 1, 0.36, 1)
        both;
}


@keyframes pipelineCardIn {

    0% {
        opacity: 0;
        transform:
            translateY(8px);
    }

    100% {
        opacity: 1;
        transform:
            translateY(0);
    }

}


/* Stagger the four stages */

.bulk-pipeline-card:nth-child(1) {
    animation-delay: 0.05s;
}

.bulk-pipeline-card:nth-child(2) {
    animation-delay: 0.15s;
}

.bulk-pipeline-card:nth-child(3) {
    animation-delay: 0.25s;
}

.bulk-pipeline-card:nth-child(4) {
    animation-delay: 0.35s;
}

/* =========================================================
   REDUCED MOTION
   ========================================================= */

@media (prefers-reduced-motion: reduce) {

    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}


footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HTML HELPERS
# ============================================================

def html_block(content):
    st.html(content)


def show_hero(
    eyebrow,
    title,
    description,
    status="DETECTION ENGINE ONLINE",
):
    html_block(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{eyebrow}</div>

            <div class="hero-title">
                <span>{title}</span>
            </div>

            <div class="hero-description">
                {description}
            </div>

            <div class="hero-status">
                <span>●</span>
                {status}
            </div>
        </div>
        """
    )


def section_header(title, subtitle=""):
    html_block(
        f"""
        <div class="section-header">
            <div class="section-header-title">
                {title}
            </div>

            <div class="section-header-subtitle">
                {subtitle}
            </div>
        </div>
        """
    )


def metric_card(
    label,
    value,
    hint,
    accent="accent-cyan",
):
    html_block(
        f"""
        <div class="metric-card {accent}">
            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-hint">
                {hint}
            </div>
        </div>
        """
    )


def panel(title, subtitle="", content=""):
    html_block(
        f"""
        <div class="info-card">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
            {content}
        </div>
        """
    )


def risk_band(probability):
    probability = float(probability)

    if probability >= 0.75:
        return "CRITICAL"

    if probability >= 0.50:
        return "HIGH"

    if probability >= 0.25:
        return "MEDIUM"

    return "LOW"


def risk_badge(probability):
    band = risk_band(probability)

    css = {
        "LOW": "risk-low",
        "MEDIUM": "risk-medium",
        "HIGH": "risk-high",
        "CRITICAL": "risk-critical",
    }[band]

    return (
        f'<span class="risk-badge {css}">'
        f"{band}"
        f"</span>"
    )


def risk_distribution(data):
    if data is None or len(data) == 0:
        return pd.DataFrame(
            {
                "Risk": [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL",
                ],
                "Transactions": [0, 0, 0, 0],
            }
        )

    bands = data["FraudProbability"].apply(risk_band)

    distribution = (
        bands
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    distribution.columns = [
        "Risk",
        "Transactions",
    ]

    return distribution


def style_chart(
    figure,
    height=420,
):
    figure.update_layout(
        height=height,

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#AEB5D0",
        ),

        margin=dict(
            l=25,
            r=25,
            t=55,
            b=25,
        ),

        title_font=dict(
            color="#F7F8FF",
            size=15,
        ),

        xaxis=dict(
            gridcolor="rgba(255,255,255,0.055)",
            zerolinecolor="rgba(255,255,255,0.055)",
        ),

        yaxis=dict(
            gridcolor="rgba(255,255,255,0.055)",
            zerolinecolor="rgba(255,255,255,0.055)",
        ),

        transition={
            "duration": 650,
            "easing": "cubic-in-out",
        },
    )

    return figure


# ============================================================
# DATA + MODEL
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():

    loaded_model = joblib.load(
        MODEL_PATH
    )

    loaded_frequency = joblib.load(
        MERCHANT_FREQ_PATH
    )

    return (
        loaded_model,
        loaded_frequency,
    )


try:

    df = load_data()

    model, merchant_frequency = load_model()

except Exception as error:

    st.error(
        "FraudGuard could not load the required "
        "dataset/model files."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# ANALYTICS DATA
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_analytics(data):

    result = data.copy()

    result["TransactionDate"] = pd.to_datetime(
        result["TransactionDate"],
        dayfirst=True,
        errors="coerce",
    )

    result["Month"] = (
        result["TransactionDate"]
        .dt.month
    )

    result["MonthName"] = (
        result["TransactionDate"]
        .dt.strftime("%b")
    )

    result["DayOfWeek"] = (
        result["TransactionDate"]
        .dt.day_name()
    )

    return result


analytics_df = prepare_analytics(df)


# ============================================================
# MODEL FEATURE ENGINEERING
# ============================================================

def create_model_features(
    data,
    frequency_map,
):

    working = data.copy()

    dates = pd.to_datetime(
        working["TransactionDate"],
        dayfirst=True,
        errors="coerce",
    )

    working["__Date"] = dates

    working["__Merchant"] = pd.to_numeric(
        working["MerchantID"],
        errors="coerce",
    )

    working["__Amount"] = pd.to_numeric(
        working["Amount"],
        errors="coerce",
    )

    working["__TransactionType"] = (
        working["TransactionType"]
        .astype(str)
        .str.lower()
    )

    working["__Location"] = (
        working["Location"]
        .astype(str)
    )

    valid_mask = (
        working["__Date"].notna()
        &
        working["__Merchant"].notna()
        &
        working["__Amount"].notna()
    )

    working = working[
        valid_mask
    ].copy()

    merchant_keys = (
        working["__Merchant"]
        .astype(int)
    )

    merchant_freq = merchant_keys.map(
        frequency_map
    ).fillna(0)

    features = pd.DataFrame(
        index=working.index
    )

    features["Amount"] = (
        working["__Amount"]
    )

    features["TransactionMonth"] = (
        working["__Date"].dt.month
    )

    features["TransactionDay"] = (
        working["__Date"].dt.day
    )

    features["TransactionDayOfWeek"] = (
        working["__Date"].dt.weekday
    )

    features["TransactionType_refund"] = (
        working["__TransactionType"]
        .eq("refund")
        .astype(int)
    )

    for location in [
        "Dallas",
        "Houston",
        "Los Angeles",
        "New York",
        "Philadelphia",
        "Phoenix",
        "San Antonio",
        "San Diego",
        "San Jose",
    ]:

        features[
            f"Location_{location}"
        ] = (
            working["__Location"]
            .eq(location)
            .astype(int)
        )

    features["MerchantFrequency"] = (
        merchant_freq.values
    )

    features = features[
        MODEL_FEATURES
    ]

    return (
        working,
        features,
        valid_mask,
    )


def score_transactions(
    data,
    frequency_map,
):

    working, features, valid_mask = (
        create_model_features(
            data,
            frequency_map,
        )
    )

    if len(features) == 0:
        raise ValueError(
            "No valid transactions could be "
            "converted into model features."
        )

    probabilities = (
        model
        .predict_proba(features)[:, 1]
    )

    result = working.copy()

    result["FraudProbability"] = (
        probabilities
    )

    result["FraudPrediction"] = (
        probabilities >= 0.5
    ).astype(int)

    result["Risk"] = (
        result["FraudPrediction"]
        .map(
            {
                0: "Legitimate",
                1: "Potential Fraud",
            }
        )
    )

    return result


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "current_page": "RISK OVERVIEW",

    "live_results": None,
    "live_feed": None,

    "investigation_transaction": None,
    "investigation_result": None,

    "bulk_scan_results": None,
    "bulk_scan_threshold": 0.50,

    "case_status": "OPEN",

    "uploaded_dataset": None,
    "scan_ready_dataset": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    "RISK OVERVIEW",
    "LIVE MONITOR",
    "INVESTIGATION",
    "BULK SCANNER",
    "INTELLIGENCE",
    "DATA EXPLORER",
    "MODEL LAB",
]


def change_page(page_name):
    st.session_state.current_page = page_name


with st.sidebar:

    html_block(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                🛡️
            </div>

            <div class="sidebar-brand-text">

                <div class="sidebar-title">
                    FraudGuard
                </div>

                <div class="sidebar-subtitle">
                    FRAUD INTELLIGENCE PLATFORM
                </div>

            </div>

        </div>

        <div class="sidebar-label">
            WORKSPACE
        </div>
        """
    )

    for page_name in PAGES:

        active = (
            st.session_state.current_page
            == page_name
        )

        st.button(
            page_name,
            key=f"nav_{page_name}",
            type=(
                "primary"
                if active
                else "secondary"
            ),
            use_container_width=True,
            on_click=change_page,
            args=(page_name,),
        )

    html_block(
        """
        <div class="sidebar-status">

            <div class="sidebar-status-title">
                <span class="status-dot"></span>
                ENGINE ONLINE
            </div>

            <div class="sidebar-status-text">
                Fraud detection engine ready.
                Transaction intelligence workspace active.
            </div>

            <div class="sidebar-status-model">
                <span>MODEL</span>
                <strong>RANDOM FOREST</strong>
            </div>

        </div>
        """
    )


page = st.session_state.current_page
# ============================================================
# RISK OVERVIEW
# ============================================================

if page == "RISK OVERVIEW":

    show_hero(
        "RISK OVERVIEW · FRAUD INTELLIGENCE",
        "FraudGuard",
        (
            "A centralized fraud intelligence workspace for "
            "monitoring transaction activity, understanding "
            "risk signals and moving suspicious transactions "
            "into investigation."
        ),
        "DETECTION ENGINE ONLINE",
    )


    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    total_transactions = len(df)

    fraud_transactions = int(
        df["IsFraud"].sum()
    )

    fraud_rate = (
        fraud_transactions
        / total_transactions
        * 100
        if total_transactions
        else 0
    )

    merchant_count = (
        df["MerchantID"]
        .nunique()
    )


    section_header(
        "REAL-TIME DATASET PULSE",
        "Reference dataset intelligence",
    )


    k1, k2, k3, k4 = st.columns(4)


    with k1:

        metric_card(
            "TRANSACTIONS",
            f"{total_transactions:,}",
            "Reference dataset volume",
            "accent-cyan",
        )


    with k2:

        metric_card(
            "FRAUD SIGNALS",
            f"{fraud_transactions:,}",
            "Observed fraud labels",
            "accent-pink",
        )


    with k3:

        metric_card(
            "FRAUD RATE",
            f"{fraud_rate:.2f}%",
            "Observed dataset rate",
            "accent-violet",
        )


    with k4:

        metric_card(
            "MERCHANTS",
            f"{merchant_count:,}",
            "Unique merchant IDs",
            "accent-lime",
        )


    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    section_header(
        "THREAT ACTIVITY",
        "Move from overview to action",
    )


    q1, q2, q3 = st.columns(3)


    with q1:

        if st.button(
            "▶  OPEN LIVE MONITOR",
            use_container_width=True,
        ):

            change_page("LIVE MONITOR")
            st.rerun()


    with q2:

        if st.button(
            "🔎  INVESTIGATE TRANSACTION",
            use_container_width=True,
        ):

            change_page("INVESTIGATION")
            st.rerun()


    with q3:

        if st.button(
            "⚡  OPEN BULK SCANNER",
            use_container_width=True,
        ):

            change_page("BULK SCANNER")
            st.rerun()


    # --------------------------------------------------------
    # LIVE ACTIVITY SNAPSHOT
    # --------------------------------------------------------

    section_header(
        "SIGNAL INTELLIGENCE",
        "Observed behavior in the reference dataset",
    )


    s1, s2 = st.columns(
        [1.35, 1]
    )


    with s1:

        monthly = (
            analytics_df
            .groupby(
                [
                    "Month",
                    "MonthName",
                ]
            )
            .agg(
                Transactions=(
                    "IsFraud",
                    "count",
                ),
                Fraud=(
                    "IsFraud",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values("Month")
        )

        monthly["FraudRate"] = (
            monthly["Fraud"]
            /
            monthly["Transactions"]
            *
            100
        )

        fig = px.area(
            monthly,
            x="MonthName",
            y="FraudRate",
            markers=True,
            title="Observed Monthly Fraud Rate",
        )

        fig.update_traces(
            line_color="#8A5CFF",
            marker_color="#39E7FF",
        )

        fig.update_yaxes(
            ticksuffix="%",
            title="Fraud Rate (%)",
        )

        st.plotly_chart(
            style_chart(
                fig,
                400,
            ),
            width="stretch",
        )


    with s2:

        location_data = (
            df
            .groupby("Location")["IsFraud"]
            .mean()
            .mul(100)
            .sort_values()
            .reset_index(
                name="FraudRate"
            )
        )

        fig = px.bar(
            location_data,
            x="FraudRate",
            y="Location",
            orientation="h",
            title="Observed Fraud Rate by Location",
        )

        fig.update_traces(
            marker_color="#39E7FF"
        )

        fig.update_xaxes(
            ticksuffix="%",
            title="Fraud Rate (%)",
        )

        st.plotly_chart(
            style_chart(
                fig,
                400,
            ),
            width="stretch",
        )


    # --------------------------------------------------------
    # TRANSACTION TYPE
    # --------------------------------------------------------

    section_header(
        "TRANSACTION TYPE SIGNALS",
        "Descriptive observations",
    )


    type_data = (
        df
        .groupby(
            "TransactionType"
        )["IsFraud"]
        .mean()
        .mul(100)
        .reset_index(
            name="FraudRate"
        )
    )


    fig = px.bar(
        type_data,
        x="TransactionType",
        y="FraudRate",
        title="Observed Fraud Rate by Transaction Type",
    )


    fig.update_traces(
        marker_color="#FF4EDB"
    )


    fig.update_yaxes(
        ticksuffix="%",
        title="Fraud Rate (%)",
    )


    st.plotly_chart(
        style_chart(
            fig,
            350,
        ),
        width="stretch",
    )


    st.caption(
        "Observed rates describe the reference dataset and "
        "do not establish that a particular location, merchant "
        "or transaction type is inherently suspicious."
    )


# ============================================================
# LIVE MONITOR
# ============================================================

elif page == "LIVE MONITOR":

    show_hero(
        "LIVE MONITOR · SIMULATED TRANSACTION STREAM",
        "Watch the signals move.",
        (
            "Replay transactions through the trained model and "
            "surface risk as the simulated stream progresses. "
            "This is a local replay of the reference dataset, "
            "not a connection to a live payment network."
        ),
        "SIMULATION ENGINE READY",
    )


    section_header(
        "REPLAY CONTROLS",
        "Configure the simulated transaction stream",
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        replay_count = st.slider(
            "Transactions",
            5,
            50,
            15,
        )


    with c2:

        replay_speed = st.select_slider(
            "Replay speed",
            options=[
                "Slow",
                "Normal",
                "Fast",
            ],
            value="Normal",
        )


    with c3:

        alert_threshold = st.slider(
            "Alert threshold",
            0.10,
            0.95,
            0.50,
            0.05,
        )


    speed_delay = {
        "Slow": 0.18,
        "Normal": 0.08,
        "Fast": 0.02,
    }


    if st.button(
        "▶  START LIVE REPLAY",
        use_container_width=True,
    ):

        sample = (
            df
            .sample(
                n=min(
                    replay_count,
                    len(df),
                ),
                random_state=None,
            )
            .copy()
        )


        try:

            scored = score_transactions(
                sample,
                merchant_frequency,
            )


            scored["Alert"] = (
                scored["FraudProbability"]
                >= alert_threshold
            )


            scored["RiskBand"] = (
                scored["FraudProbability"]
                .apply(risk_band)
            )


            scored = scored.reset_index(
                drop=True
            )


            st.session_state.live_results = (
                scored
            )

            st.session_state.live_feed = (
                scored.iloc[0:0]
                .copy()
            )

            st.session_state.live_speed = (
                replay_speed
            )

            st.session_state.live_threshold = (
                alert_threshold
            )

            st.session_state.live_index = 0


        except Exception as error:

            st.error(
                f"Replay could not start: {error}"
            )


    # --------------------------------------------------------
    # STREAM REPLAY
    # --------------------------------------------------------

    if (
        st.session_state.live_results
        is not None
    ):

        replay = (
            st.session_state.live_results
        )


        if st.button(
            "⚡  STREAM NEXT TRANSACTION",
            use_container_width=True,
        ):

            current_index = (
                st.session_state.get(
                    "live_index",
                    0,
                )
            )


            if current_index < len(replay):

                current_row = (
                    replay
                    .iloc[
                        current_index:
                        current_index + 1
                    ]
                    .copy()
                )


                st.session_state.live_feed = (
                    pd.concat(
                        [
                            st.session_state.live_feed,
                            current_row,
                        ],
                        ignore_index=True,
                    )
                )


                st.session_state.live_index = (
                    current_index + 1
                )


                row = current_row.iloc[0]

                probability = float(
                    row["FraudProbability"]
                )


                if probability >= 0.75:

                    st.toast(
                        "🚨 CRITICAL RISK DETECTED",
                        icon="🚨",
                    )

                elif probability >= 0.50:

                    st.toast(
                        "⚠️ HIGH RISK TRANSACTION",
                        icon="⚠️",
                    )

                else:

                    st.toast(
                        "Transaction processed",
                        icon="✅",
                    )

                st.rerun()


        feed = (
            st.session_state.live_feed
        )


        processed = len(feed)

        alerts = int(
            feed["Alert"].sum()
        ) if len(feed) else 0

        high_risk = int(
            (
                feed["FraudProbability"]
                >= 0.50
            ).sum()
        ) if len(feed) else 0

        max_risk = (
            feed["FraudProbability"].max()
            if len(feed)
            else 0
        )


        section_header(
            "LIVE THREAT ACTIVITY",
            (
                f"{processed}/{len(replay)} transactions processed"
            ),
        )


        m1, m2, m3, m4 = st.columns(4)


        with m1:

            metric_card(
                "PROCESSED",
                f"{processed:,}",
                "Stream activity",
                "accent-cyan",
            )


        with m2:

            metric_card(
                "ALERTS",
                f"{alerts:,}",
                "Above configured threshold",
                "accent-pink",
            )


        with m3:

            metric_card(
                "HIGH RISK",
                f"{high_risk:,}",
                "Probability ≥ 50%",
                "accent-violet",
            )


        with m4:

            metric_card(
                "MAX RISK",
                f"{max_risk * 100:.1f}%",
                "Current stream maximum",
                "accent-lime",
            )


        # ----------------------------------------------------
        # CURRENT ALERT
        # ----------------------------------------------------

        if len(feed):

            latest = feed.iloc[-1]

            probability = float(
                latest["FraudProbability"]
            )

            band = risk_band(
                probability
            )


            if band in [
                "HIGH",
                "CRITICAL",
            ]:

                html_block(
                    f"""
                    <div class="info-card"
                         style="
                         border-color:
                         rgba(255,78,219,0.25);
                         box-shadow:
                         0 0 35px
                         rgba(255,78,219,0.07);
                         ">

                        <div class="panel-title">
                            🚨 ACTIVE RISK SIGNAL
                        </div>

                        <div style="
                            margin-top:10px;
                            color:#FFB0E9;
                            font-size:12px;
                            font-weight:800;
                            ">

                            Transaction
                            {latest.get(
                                'TransactionID',
                                'N/A'
                            )}

                            ·

                            {band}

                            ·

                            {probability * 100:.1f}%

                        </div>

                        <div style="
                            margin-top:7px;
                            color:#8D95B1;
                            font-size:9px;
                            ">

                            Amount:
                            ${float(
                                latest['Amount']
                            ):,.2f}

                            &nbsp;&nbsp;·&nbsp;&nbsp;

                            Location:
                            {latest['Location']}

                            &nbsp;&nbsp;·&nbsp;&nbsp;

                            Merchant:
                            {latest['MerchantID']}

                        </div>

                    </div>
                    """
                )


        # ----------------------------------------------------
        # FEED TABLE
        # ----------------------------------------------------

        if len(feed):

            display = feed.copy()

            display["FraudProbability"] = (
                display["FraudProbability"]
                .mul(100)
                .round(2)
                .astype(str)
                + "%"
            )


            columns = [
                column
                for column in [
                    "TransactionID",
                    "Amount",
                    "MerchantID",
                    "TransactionDate",
                    "TransactionType",
                    "Location",
                    "FraudProbability",
                    "RiskBand",
                    "Alert",
                ]
                if column in display.columns
            ]


            st.dataframe(
                display[columns],
                width="stretch",
                height=430,
                hide_index=True,
            )


        else:

            html_block(
                """
                <div class="info-card"
                     style="
                     text-align:center;
                     padding:35px;
                     ">

                    <div style="
                        color:#39E7FF;
                        font-size:20px;
                        ">

                        ◉

                    </div>

                    <div style="
                        margin-top:10px;
                        color:#E8EBFA;
                        font-weight:900;
                        ">

                        Stream ready

                    </div>

                    <div style="
                        margin-top:5px;
                        color:#69718E;
                        font-size:9px;
                        ">

                        Start the replay and process
                        transactions one by one.

                    </div>

                </div>
                """
            )


        # ----------------------------------------------------
        # FINISH REPLAY
        # ----------------------------------------------------

        if processed < len(replay):

            if st.button(
                "⏩  PROCESS REMAINING TRANSACTIONS",
                use_container_width=True,
            ):

                remaining = replay.iloc[
                    processed:
                ].copy()


                st.session_state.live_feed = (
                    pd.concat(
                        [
                            feed,
                            remaining,
                        ],
                        ignore_index=True,
                    )
                )


                st.session_state.live_index = (
                    len(replay)
                )


                st.toast(
                    "Replay complete",
                    icon="✅",
                )

                st.rerun()


        else:

            st.success(
                "Simulation complete. All selected "
                "transactions have been processed."
            )
            # ============================================================
# INVESTIGATION
# ============================================================

elif page == "INVESTIGATION":

    show_hero(
        "INVESTIGATION · CASE WORKSPACE",
        "Interrogate the risk.",
        (
            "Review an individual transaction using the same "
            "feature-engineering and Random Forest prediction "
            "pipeline used throughout FraudGuard."
        ),
        "ANALYSIS ENGINE READY",
    )


    # --------------------------------------------------------
    # PRELOADED TRANSACTION
    # --------------------------------------------------------

    preloaded = (
        st.session_state
        .get(
            "investigation_transaction"
        )
    )


    section_header(
        "TRANSACTION INPUT",
        "Model features are engineered automatically",
    )


    if preloaded is not None:

        st.info(
            "A transaction was transferred into Investigation "
            "from another FraudGuard workflow."
        )


    c1, c2 = st.columns(2)


    with c1:

        default_amount = (
            float(
                preloaded["Amount"]
            )
            if preloaded is not None
            and "Amount" in preloaded
            else 1000.0
        )


        default_merchant = (
            int(
                preloaded["MerchantID"]
            )
            if preloaded is not None
            and "MerchantID" in preloaded
            else 1
        )


        amount = st.number_input(
            "Transaction Amount (USD)",
            min_value=0.0,
            value=default_amount,
            step=10.0,
        )


        merchant_id = st.number_input(
            "Merchant ID",
            min_value=1,
            value=default_merchant,
            step=1,
        )


    with c2:

        default_date = date.today()


        if preloaded is not None:

            try:

                parsed = pd.to_datetime(
                    preloaded[
                        "TransactionDate"
                    ],
                    dayfirst=True,
                )

                if not pd.isna(parsed):

                    default_date = (
                        parsed.date()
                    )

            except Exception:
                pass


        transaction_date = st.date_input(
            "Transaction Date",
            value=default_date,
        )


        default_type = (
            str(
                preloaded[
                    "TransactionType"
                ]
            ).lower()
            if preloaded is not None
            and "TransactionType"
            in preloaded
            else "purchase"
        )


        transaction_type = st.selectbox(
            "Transaction Type",
            TRANSACTION_TYPES,
            index=(
                TRANSACTION_TYPES.index(
                    default_type
                )
                if default_type
                in TRANSACTION_TYPES
                else 0
            ),
        )


    default_location = (
        str(
            preloaded["Location"]
        )
        if preloaded is not None
        and "Location" in preloaded
        else "Chicago"
    )


    location = st.selectbox(
        "Transaction Location",
        LOCATIONS,
        index=(
            LOCATIONS.index(
                default_location
            )
            if default_location in LOCATIONS
            else 0
        ),
    )


    if st.button(
        "🔎  ANALYZE TRANSACTION",
        use_container_width=True,
    ):

        transaction = pd.DataFrame(
            [
                {
                    "TransactionID": (
                        preloaded.get(
                            "TransactionID"
                        )
                        if preloaded is not None
                        else None
                    ),

                    "Amount": amount,

                    "MerchantID": merchant_id,

                    "TransactionDate": (
                        transaction_date
                    ),

                    "TransactionType": (
                        transaction_type
                    ),

                    "Location": location,
                }
            ]
        )


        try:

            scored = score_transactions(
                transaction,
                merchant_frequency,
            )


            probability = float(
                scored.iloc[0][
                    "FraudProbability"
                ]
            )


            prediction = (
                probability >= 0.5
            )


            band = risk_band(
                probability
            )


            st.session_state[
                "investigation_result"
            ] = scored.iloc[0].to_dict()


            st.session_state[
                "investigation_transaction"
            ] = transaction.iloc[0].to_dict()


            st.session_state[
                "case_status"
            ] = "OPEN"


            st.toast(
                "Investigation case created",
                icon="🔎",
            )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


    # --------------------------------------------------------
    # CASE RESULT
    # --------------------------------------------------------

    result = st.session_state.get(
        "investigation_result"
    )


    if result is not None:

        probability = float(
            result["FraudProbability"]
        )

        band = risk_band(
            probability
        )

        prediction = (
            probability >= 0.5
        )


        section_header(
            "INVESTIGATION CASE",
            "Machine-learning decision support",
        )


        transaction_id = (
            result.get(
                "TransactionID"
            )
        )


        if (
            transaction_id
            is None
            or pd.isna(transaction_id)
        ):

            transaction_id = (
                "FG-"
                +
                uuid.uuid4()
                .hex[:8]
                .upper()
            )


        case_status = (
            st.session_state
            .get(
                "case_status",
                "OPEN",
            )
        )


        # ----------------------------------------------------
        # CASE HEADER
        # ----------------------------------------------------

        html_block(
            f"""
            <div class="info-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    ">

                    <div>

                        <div style="
                            color:#6F7897;
                            font-size:8px;
                            font-weight:900;
                            letter-spacing:.12em;
                            ">

                            CASE ID

                        </div>

                        <div style="
                            color:#FFFFFF;
                            font-size:22px;
                            font-weight:950;
                            margin-top:5px;
                            ">

                            {transaction_id}

                        </div>

                    </div>

                    <div>

                        <span class="risk-badge
                        {
                            'risk-critical'
                            if band == 'CRITICAL'
                            else
                            'risk-high'
                            if band == 'HIGH'
                            else
                            'risk-medium'
                            if band == 'MEDIUM'
                            else
                            'risk-low'
                        }">

                            {band}

                        </span>

                        <div style="
                            color:#6E7795;
                            font-size:8px;
                            margin-top:7px;
                            text-align:right;
                            ">

                            CASE STATUS:
                            {case_status}

                        </div>

                    </div>

                </div>

            </div>
            """
        )


        # ----------------------------------------------------
        # ANIMATED RISK SCORE
        # ----------------------------------------------------

        section_header(
            "MODEL ASSESSMENT",
            "Random Forest output",
        )


        r1, r2 = st.columns(
            [1.2, 1]
        )


        with r1:

            html_block(
                f"""
                <div class="info-card"
                     style="
                     text-align:center;
                     padding:28px;
                     ">

                    <div style="
                        color:#6D7594;
                        font-size:8px;
                        font-weight:900;
                        letter-spacing:.14em;
                        ">

                        FRAUD PROBABILITY

                    </div>

                    <div style="
                        color:#FFFFFF;
                        font-size:52px;
                        font-weight:950;
                        letter-spacing:-.06em;
                        margin-top:10px;
                        ">

                        {probability * 100:.2f}%

                    </div>

                    <div style="
                        margin-top:10px;
                        color:
                        {
                            '#FF4EDB'
                            if band in ['HIGH','CRITICAL']
                            else '#B7FF63'
                        };
                        font-size:11px;
                        font-weight:900;
                        letter-spacing:.12em;
                        ">

                        {band} RISK

                    </div>

                    <div style="
                        margin-top:17px;
                        height:7px;
                        border-radius:999px;
                        background:#191D37;
                        overflow:hidden;
                        ">

                        <div style="
                            width:{probability * 100:.2f}%;
                            height:100%;
                            border-radius:999px;
                            background:
                            linear-gradient(
                                90deg,
                                #39E7FF,
                                #8A5CFF,
                                #FF4EDB
                            );
                            box-shadow:
                            0 0 18px
                            rgba(138,92,255,.35);
                            "></div>

                    </div>

                    <div style="
                        margin-top:10px;
                        color:#69718E;
                        font-size:8px;
                        ">

                        Decision threshold:
                        50%

                    </div>

                </div>
                """
            )


        with r2:

            merchant_frequency_value = (
                merchant_frequency.get(
                    int(merchant_id),
                    0,
                )
            )


            metric_card(
                "MERCHANT FREQUENCY",
                f"{int(merchant_frequency_value):,}",
                "Training-data frequency",
                "accent-cyan",
            )


            metric_card(
                "MODEL DECISION",
                (
                    "REVIEW"
                    if prediction
                    else "BELOW THRESHOLD"
                ),
                "Configured threshold: 50%",
                (
                    "accent-pink"
                    if prediction
                    else "accent-lime"
                ),
            )


        # ----------------------------------------------------
        # INPUT SIGNALS
        # ----------------------------------------------------

        section_header(
            "MODEL INPUT SIGNALS",
            "Inputs supplied to the prediction pipeline",
        )


        sig1, sig2, sig3, sig4, sig5 = (
            st.columns(5)
        )


        with sig1:

            metric_card(
                "AMOUNT",
                f"${amount:,.2f}",
                "Transaction value",
                "accent-cyan",
            )


        with sig2:

            metric_card(
                "MERCHANT",
                str(merchant_id),
                "Merchant ID",
                "accent-violet",
            )


        with sig3:

            metric_card(
                "TYPE",
                transaction_type.upper(),
                "Transaction type",
                "accent-pink",
            )


        with sig4:

            metric_card(
                "LOCATION",
                location.upper(),
                "Transaction location",
                "accent-lime",
            )


        with sig5:

            metric_card(
                "DATE",
                transaction_date.strftime(
                    "%d %b"
                ),
                "Transaction date",
                "accent-yellow",
            )


        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        section_header(
            "CASE ACTIONS",
            "Investigation workflow",
        )


        a1, a2, a3 = st.columns(3)


        with a1:

            if st.button(
                "⚠  MARK FOR REVIEW",
                use_container_width=True,
            ):

                st.session_state.case_status = (
                    "REVIEW REQUIRED"
                )

                st.toast(
                    "Case marked for review",
                    icon="⚠️",
                )

                st.rerun()


        with a2:

            if st.button(
                "✓  CLEAR CASE",
                use_container_width=True,
            ):

                st.session_state.case_status = (
                    "CLEARED"
                )

                st.toast(
                    "Case marked as cleared",
                    icon="✅",
                )

                st.rerun()


        with a3:

            case_data = pd.DataFrame(
                [
                    {
                        "CaseID": transaction_id,
                        "Amount": amount,
                        "MerchantID": merchant_id,
                        "Date": transaction_date,
                        "Type": transaction_type,
                        "Location": location,
                        "FraudProbability": (
                            probability
                        ),
                        "RiskBand": band,
                        "Status": (
                            st.session_state.case_status
                        ),
                    }
                ]
            )


            st.download_button(
                "⬇  EXPORT CASE",
                data=case_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"{transaction_id}_case.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


        # ----------------------------------------------------
        # ENGINEERED FEATURES
        # ----------------------------------------------------

        with st.expander(
            "VIEW ENGINEERED MODEL INPUT"
        ):

            transaction = pd.DataFrame(
                [
                    {
                        "Amount": amount,
                        "MerchantID": merchant_id,
                        "TransactionDate": (
                            transaction_date
                        ),
                        "TransactionType": (
                            transaction_type
                        ),
                        "Location": location,
                    }
                ]
            )


            _, engineered, _ = (
                create_model_features(
                    transaction,
                    merchant_frequency,
                )
            )


            st.dataframe(
                engineered,
                width="stretch",
                hide_index=True,
            )


        st.caption(
            "The model input signals above describe the "
            "information supplied to the model. They are not "
            "individual feature-attribution scores."
        )
        # ============================================================
# BULK SCANNER
# ============================================================

elif page == "BULK SCANNER":

    show_hero(
        "BULK SCANNER · DATA PIPELINE",
        "Scan the dataset.",
        (
            "Upload transaction data, map the required fields, "
            "engineer the model inputs and process the dataset "
            "through the FraudGuard detection pipeline."
        ),
        "CSV SCANNER READY",
    )


    # --------------------------------------------------------
    # PIPELINE STATUS
    # --------------------------------------------------------

    section_header(
        "SCAN PIPELINE",
        "Four-stage transaction intelligence workflow",
    )


    p1, p2, p3, p4 = st.columns(4)


    pipeline_cards = [
        (
            p1,
            "01",
            "DATA INGESTION",
            "Upload and validate",
        ),
        (
            p2,
            "02",
            "FEATURE ENGINEERING",
            "Prepare model inputs",
        ),
        (
            p3,
            "03",
            "MODEL SCAN",
            "Run Random Forest",
        ),
        (
            p4,
            "04",
            "RESULTS",
            "Review risk signals",
        ),
    ]


    for col, number, title, subtitle in (
        pipeline_cards
    ):

        with col:

            html_block(
                f"""
                <div class="info-card bulk-pipeline-card"
                     style="
                     min-height:135px;
                     ">

                    <div style="
                        color:#39E7FF;
                        font-family:Orbitron;
                        font-size:17px;
                        font-weight:800;
                        ">

                        {number}

                    </div>

                    <div style="
                        margin-top:12px;
                        color:#F4F6FF;
                        font-size:14px;
                        font-weight:900;
                        ">

                        {title}

                    </div>

                    <div style="
                        margin-top:7px;
                        color:#9AA4C2;
                        font-size:12px;
                        ">

                        {subtitle}

                    </div>

                </div>
                """
            )


    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    section_header(
        "01 · DATA INGESTION",
        "CSV transaction intake",
    )


    uploaded_file = st.file_uploader(
        "Drop your transaction CSV here",
        type=["csv"],
        label_visibility="collapsed",
    )


    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.session_state.uploaded_dataset = (
                uploaded_df
            )

        except Exception as error:

            st.error(
                f"Could not read CSV: {error}"
            )


    uploaded = (
        st.session_state.uploaded_dataset
    )


    if uploaded is not None:

        st.success(
            f"CSV loaded successfully — "
            f"{len(uploaded):,} rows · "
            f"{uploaded.shape[1]} columns"
        )


        required = [
            "Amount",
            "MerchantID",
            "TransactionDate",
            "TransactionType",
            "Location",
        ]


        missing = [
            column
            for column in required
            if column not in uploaded.columns
        ]


        # ----------------------------------------------------
        # AUTOMATIC RECOGNITION
        # ----------------------------------------------------

        if not missing:

            html_block(
                """
                <div class="info-card"
                     style="
                     border-color:
                     rgba(183,255,99,.18);
                     ">

                    <div style="
                        color:#B7FF63;
                        font-weight:900;
                        font-size:10px;
                        ">

                        ✓ REQUIRED FIELDS DETECTED

                    </div>

                    <div style="
                        margin-top:6px;
                        color:#69718E;
                        font-size:8px;
                        ">

                        Amount · MerchantID · Date ·
                        TransactionType · Location

                    </div>

                </div>
                """
            )


            st.session_state.scan_ready_dataset = (
                uploaded
            )


        else:

            st.warning(
                "Some required fields need manual mapping."
            )


            available = [
                "—"
            ] + list(
                uploaded.columns
            )


            m1, m2 = st.columns(2)


            with m1:

                map_amount = st.selectbox(
                    "Amount",
                    available,
                    key="bulk_amount",
                )

                map_merchant = st.selectbox(
                    "Merchant ID",
                    available,
                    key="bulk_merchant",
                )

                map_date = st.selectbox(
                    "Transaction Date",
                    available,
                    key="bulk_date",
                )


            with m2:

                map_type = st.selectbox(
                    "Transaction Type",
                    available,
                    key="bulk_type",
                )

                map_location = st.selectbox(
                    "Location",
                    available,
                    key="bulk_location",
                )


            if st.button(
                "✓  APPLY COLUMN MAPPING",
                use_container_width=True,
            ):

                selections = {
                    "Amount": map_amount,
                    "MerchantID": map_merchant,
                    "TransactionDate": map_date,
                    "TransactionType": map_type,
                    "Location": map_location,
                }


                mapping = {
                    source: target
                    for target, source
                    in selections.items()
                    if source != "—"
                }


                normalized = uploaded.rename(
                    columns=mapping
                )


                still_missing = [
                    column
                    for column in required
                    if column not in normalized.columns
                ]


                if still_missing:

                    st.error(
                        "Still missing: "
                        +
                        ", ".join(
                            still_missing
                        )
                    )

                else:

                    st.session_state.scan_ready_dataset = (
                        normalized
                    )

                    st.success(
                        "Column mapping applied."
                    )

                    st.rerun()


    # --------------------------------------------------------
    # READY DATASET
    # --------------------------------------------------------

    scan_dataset = (
        st.session_state.scan_ready_dataset
    )


    if scan_dataset is not None:

        section_header(
            "02 · FEATURE ENGINEERING",
            f"{len(scan_dataset):,} rows ready",
        )


        s1, s2 = st.columns(2)


        with s1:

            threshold = st.slider(
                "Fraud Alert Threshold",
                0.10,
                0.95,
                0.50,
                0.05,
                key="bulk_threshold",
            )


        with s2:

            max_rows = st.number_input(
                "Rows to scan",
                min_value=1,
                max_value=max(
                    1,
                    len(scan_dataset),
                ),
                value=min(
                    10000,
                    len(scan_dataset),
                ),
                step=100,
                key="bulk_rows",
            )


        if st.button(
            "⚡  START FRAUDGUARD PIPELINE",
            use_container_width=True,
        ):

            rows = (
                scan_dataset
                .head(
                    int(max_rows)
                )
                .copy()
            )


            progress = st.progress(
                0,
                text="Starting pipeline...",
            )


            status_box = st.empty()


            try:

                # --------------------------------------------
                # STAGE 1
                # --------------------------------------------

                progress.progress(
                    15,
                    text="01 · Ingesting transaction data...",
                )

                status_box.info(
                    "✓ File parsed · "
                    f"{len(rows):,} rows selected"
                )


                time.sleep(0.25)


                # --------------------------------------------
                # STAGE 2
                # --------------------------------------------

                progress.progress(
                    40,
                    text="02 · Engineering model features...",
                )

                status_box.info(
                    "✓ Dates parsed · "
                    "✓ transaction type encoded · "
                    "✓ location features generated"
                )


                time.sleep(0.25)


                # --------------------------------------------
                # MODEL
                # --------------------------------------------

                results = score_transactions(
                    rows,
                    merchant_frequency,
                )


                progress.progress(
                    70,
                    text="03 · Running Random Forest detection...",
                )


                results["Alert"] = (
                    results["FraudProbability"]
                    >= threshold
                )


                results["RiskBand"] = (
                    results["FraudProbability"]
                    .apply(risk_band)
                )


                results = (
                    results
                    .sort_values(
                        "FraudProbability",
                        ascending=False,
                    )
                    .reset_index(drop=True)
                )


                time.sleep(0.25)


                # --------------------------------------------
                # RESULTS
                # --------------------------------------------

                progress.progress(
                    100,
                    text="04 · Scan complete.",
                )


                status_box.success(
                    "✓ Pipeline completed successfully."
                )


                st.session_state.bulk_scan_results = (
                    results
                )

                st.session_state.bulk_scan_threshold = (
                    threshold
                )


                st.toast(
                    "Bulk scan complete",
                    icon="✅",
                )


            except Exception as error:

                progress.empty()

                status_box.empty()

                st.error(
                    f"Bulk scan failed: {error}"
                )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = (
        st.session_state.bulk_scan_results
    )


    if results is not None:

        section_header(
            "04 · SCAN RESULTS",
            "Model-generated risk intelligence",
        )


        alerts = int(
            results["Alert"].sum()
        )


        alert_rate = (
            alerts
            /
            len(results)
            *
            100
            if len(results)
            else 0
        )


        max_risk = (
            results[
                "FraudProbability"
            ].max()
            if len(results)
            else 0
        )


        critical = int(
            (
                results["RiskBand"]
                == "CRITICAL"
            ).sum()
        )


        b1, b2, b3, b4 = st.columns(4)


        with b1:

            metric_card(
                "ROWS SCANNED",
                f"{len(results):,}",
                "Processed transactions",
                "accent-cyan",
            )


        with b2:

            metric_card(
                "ALERTS",
                f"{alerts:,}",
                "Above selected threshold",
                "accent-pink",
            )


        with b3:

            metric_card(
                "ALERT RATE",
                f"{alert_rate:.2f}%",
                "Share flagged",
                "accent-violet",
            )


        with b4:

            metric_card(
                "CRITICAL",
                f"{critical:,}",
                "Probability ≥ 75%",
                "accent-lime",
            )


        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        d1, d2 = st.columns(
            [1, 1.4]
        )


        with d1:

            distribution = risk_distribution(
                results
            )


            fig = px.bar(
                distribution,
                x="Risk",
                y="Transactions",
                title="Risk Distribution",
            )


            fig.update_traces(
                marker_color="#8A5CFF"
            )


            st.plotly_chart(
                style_chart(
                    fig,
                    320,
                ),
                width="stretch",
            )


        with d2:

            result_filter = st.segmented_control(
                "Results",
                [
                    "ALL",
                    "ALERTS",
                    "HIGH RISK",
                    "CRITICAL",
                    "LOW RISK",
                ],
                default="ALL",
            )


            filtered_results = results.copy()


            if result_filter == "ALERTS":

                filtered_results = (
                    filtered_results[
                        filtered_results["Alert"]
                    ]
                )


            elif result_filter == "HIGH RISK":

                filtered_results = (
                    filtered_results[
                        filtered_results[
                            "FraudProbability"
                        ] >= 0.50
                    ]
                )


            elif result_filter == "CRITICAL":

                filtered_results = (
                    filtered_results[
                        filtered_results[
                            "FraudProbability"
                        ] >= 0.75
                    ]
                )


            elif result_filter == "LOW RISK":

                filtered_results = (
                    filtered_results[
                        filtered_results[
                            "FraudProbability"
                        ] < 0.25
                    ]
                )


            display = filtered_results.copy()


            if len(display):

                display["FraudProbability"] = (
                    display[
                        "FraudProbability"
                    ]
                    .mul(100)
                    .round(2)
                    .astype(str)
                    + "%"
                )


            columns = [
                column
                for column in [
                    "TransactionID",
                    "Amount",
                    "MerchantID",
                    "TransactionDate",
                    "TransactionType",
                    "Location",
                    "FraudProbability",
                    "RiskBand",
                    "Alert",
                ]
                if column in display.columns
            ]


            st.dataframe(
                display[columns].head(500),
                width="stretch",
                height=320,
                hide_index=True,
            )


        # ----------------------------------------------------
        # INVESTIGATION BRIDGE
        # ----------------------------------------------------

        if len(filtered_results):

            section_header(
                "INVESTIGATION BRIDGE",
                "Transfer a selected transaction into Case Review",
            )


            transaction_options = []


            for idx, row in (
                filtered_results
                .head(100)
                .iterrows()
            ):

                transaction_options.append(
                    (
                        idx,
                        (
                            f"{row.get('TransactionID', 'TX')}"
                            f" · "
                            f"{row['RiskBand']}"
                            f" · "
                            f"{row['FraudProbability'] * 100:.1f}%"
                            f" · "
                            f"{row['Location']}"
                        ),
                    )
                )


            selected = st.selectbox(
                "Select transaction",
                transaction_options,
                format_func=lambda x: x[1],
            )


            if st.button(
                "🔎  INVESTIGATE SELECTED TRANSACTION",
                use_container_width=True,
            ):

                selected_index = selected[0]

                selected_row = (
                    filtered_results
                    .loc[selected_index]
                )


                st.session_state[
                    "investigation_transaction"
                ] = selected_row.to_dict()


                st.session_state[
                    "investigation_result"
                ] = None


                st.session_state[
                    "case_status"
                ] = "OPEN"


                st.session_state.current_page = (
                    "INVESTIGATION"
                )


                st.toast(
                    "Transaction transferred to Investigation",
                    icon="🔎",
                )

                st.rerun()


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_output = (
            results
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )


        st.download_button(
            "⬇  DOWNLOAD COMPLETE SCAN RESULTS",
            data=csv_output,
            file_name="fraudguard_scan_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
# ============================================================
# INTELLIGENCE
# ============================================================

elif page == "INTELLIGENCE":

    show_hero(
        "INTELLIGENCE · SIGNAL EXPLORATION",
        "Find the signal.",
        (
            "Explore fraud behavior using interactive filters "
            "across location, transaction type, date and "
            "transaction amount."
        ),
        "ANALYTICS ENGINE READY",
    )

    # --------------------------------------------------------
    # INTELLIGENCE FILTERS
    # --------------------------------------------------------

    section_header(
        "INTELLIGENCE FILTERS",
        "All observations update from the filtered dataset",
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        intelligence_location = st.selectbox(
            "Location",
            ["All"]
            +
            sorted(
                analytics_df[
                    "Location"
                ]
                .dropna()
                .unique()
                .tolist()
            ),
            key="intel_location",
        )

    with f2:

        intelligence_type = st.selectbox(
            "Transaction Type",
            ["All"]
            +
            sorted(
                analytics_df[
                    "TransactionType"
                ]
                .dropna()
                .unique()
                .tolist()
            ),
            key="intel_type",
        )

    with f3:

        min_amount = float(
            analytics_df[
                "Amount"
            ].min()
        )

        max_amount = float(
            analytics_df[
                "Amount"
            ].max()
        )

        amount_range = st.slider(
            "Amount Range",
            min_value=min_amount,
            max_value=max_amount,
            value=(
                min_amount,
                max_amount,
            ),
            key="intel_amount",
        )

    date_values = (
        analytics_df[
            "TransactionDate"
        ]
        .dropna()
    )

    if len(date_values):

        min_date = (
            date_values.min()
            .date()
        )

        max_date = (
            date_values.max()
            .date()
        )

    else:

        min_date = date.today()
        max_date = date.today()

    date_range = st.date_input(
        "Transaction Date Range",
        value=(
            min_date,
            max_date,
        ),
        key="intel_date",
    )

    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    intelligence_df = analytics_df.copy()

    if intelligence_location != "All":

        intelligence_df = (
            intelligence_df[
                intelligence_df[
                    "Location"
                ]
                ==
                intelligence_location
            ]
        )

    if intelligence_type != "All":

        intelligence_df = (
            intelligence_df[
                intelligence_df[
                    "TransactionType"
                ]
                ==
                intelligence_type
            ]
        )

    intelligence_df = (
        intelligence_df[
            (
                intelligence_df["Amount"]
                >= amount_range[0]
            )
            &
            (
                intelligence_df["Amount"]
                <= amount_range[1]
            )
        ]
    )

    if isinstance(
        date_range,
        tuple,
    ) and len(date_range) == 2:

        intelligence_df = (
            intelligence_df[
                (
                    intelligence_df[
                        "TransactionDate"
                    ].dt.date
                    >= date_range[0]
                )
                &
                (
                    intelligence_df[
                        "TransactionDate"
                    ].dt.date
                    <= date_range[1]
                )
            ]
        )

    # --------------------------------------------------------
    # INTELLIGENCE SNAPSHOT
    # --------------------------------------------------------

    filtered_count = len(
        intelligence_df
    )

    filtered_fraud = int(
        intelligence_df[
            "IsFraud"
        ].sum()
    ) if filtered_count else 0

    filtered_rate = (
        filtered_fraud
        /
        filtered_count
        *
        100
        if filtered_count
        else 0
    )

    filtered_amount = (
        intelligence_df[
            "Amount"
        ].mean()
        if filtered_count
        else 0
    )

    section_header(
        "INTELLIGENCE SNAPSHOT",
        "Current filtered dataset",
    )

    i1, i2, i3, i4 = st.columns(4)

    with i1:

        metric_card(
            "FILTERED TRANSACTIONS",
            f"{filtered_count:,}",
            "Rows matching filters",
            "accent-cyan",
        )

    with i2:

        metric_card(
            "OBSERVED FRAUD",
            f"{filtered_fraud:,}",
            "Fraud labels in selection",
            "accent-pink",
        )

    with i3:

        metric_card(
            "OBSERVED RATE",
            f"{filtered_rate:.2f}%",
            "Fraud rate in selection",
            "accent-violet",
        )

    with i4:

        metric_card(
            "AVG AMOUNT",
            f"${filtered_amount:,.2f}",
            "Mean transaction amount",
            "accent-lime",
        )

    # --------------------------------------------------------
    # ANALYTICS LENS
    # --------------------------------------------------------

    section_header(
        "ANALYTICS LENS",
        "Choose one signal",
    )

    intelligence_view = st.segmented_control(
        "Analytics",
        [
            "Monthly Trend",
            "Merchant Signals",
            "Location Signals",
            "Transaction Type",
            "Day of Week",
            "Amount Distribution",
        ],
        default="Monthly Trend",
        label_visibility="collapsed",
        key="intelligence_view",
    )

    # --------------------------------------------------------
    # ANALYTICS
    # --------------------------------------------------------

    if filtered_count == 0:

        st.warning(
            "No transactions match the selected filters."
        )

    else:

        # ----------------------------------------------------
        # MONTHLY TREND
        # ----------------------------------------------------

        if intelligence_view == "Monthly Trend":

            monthly = (
                intelligence_df
                .groupby(
                    [
                        "Month",
                        "MonthName",
                    ]
                )
                .agg(
                    Transactions=(
                        "IsFraud",
                        "count",
                    ),
                    Fraud=(
                        "IsFraud",
                        "sum",
                    ),
                )
                .reset_index()
                .sort_values("Month")
            )

            monthly["FraudRate"] = (
                monthly["Fraud"]
                /
                monthly["Transactions"]
                *
                100
            )

            fig = px.area(
                monthly,
                x="MonthName",
                y="FraudRate",
                markers=True,
                title="Filtered Monthly Fraud Rate",
            )

            fig.update_traces(
                line_color="#8A5CFF",
                marker_color="#39E7FF",
            )

            fig.update_yaxes(
                ticksuffix="%",
                title="Fraud Rate (%)",
            )

            st.plotly_chart(
                style_chart(
                    fig,
                    470,
                ),
                width="stretch",
            )

        # ----------------------------------------------------
        # MERCHANT SIGNALS
        # ----------------------------------------------------

        elif intelligence_view == "Merchant Signals":

            merchant_data = (
                intelligence_df
                .groupby("MerchantID")
                .agg(
                    Transactions=(
                        "IsFraud",
                        "count",
                    ),
                    Fraud=(
                        "IsFraud",
                        "sum",
                    ),
                )
                .reset_index()
            )

            merchant_data["FraudRate"] = (
                merchant_data["Fraud"]
                /
                merchant_data["Transactions"]
                *
                100
            )

            merchant_data = (
                merchant_data
                .sort_values(
                    [
                        "FraudRate",
                        "Transactions",
                    ],
                    ascending=False,
                )
                .head(20)
            )

            fig = px.bar(
                merchant_data,
                x="MerchantID",
                y="FraudRate",
                title="Highest observed merchant fraud-rate signals",
            )

            fig.update_traces(
                marker_color="#FF4EDB"
            )

            fig.update_yaxes(
                ticksuffix="%",
                title="Fraud Rate (%)",
            )

            st.plotly_chart(
                style_chart(fig),
                width="stretch",
            )

            st.caption(
                "These are descriptive observations from "
                "the selected data and are not proof of "
                "wrongdoing."
            )

        # ----------------------------------------------------
        # LOCATION SIGNALS
        # ----------------------------------------------------

        elif intelligence_view == "Location Signals":

            location_data = (
                intelligence_df
                .groupby("Location")["IsFraud"]
                .mean()
                .mul(100)
                .sort_values()
                .reset_index(
                    name="FraudRate"
                )
            )

            fig = px.bar(
                location_data,
                x="FraudRate",
                y="Location",
                orientation="h",
                title="Observed Fraud Rate by Location",
            )

            fig.update_traces(
                marker_color="#39E7FF"
            )

            fig.update_xaxes(
                ticksuffix="%",
                title="Fraud Rate (%)",
            )

            st.plotly_chart(
                style_chart(fig),
                width="stretch",
            )

            st.caption(
                "Location is shown as an observed dataset "
                "dimension, not as evidence that transactions "
                "from a location are inherently suspicious."
            )

        # ----------------------------------------------------
        # TRANSACTION TYPE
        # ----------------------------------------------------

        elif intelligence_view == "Transaction Type":

            type_data = (
                intelligence_df
                .groupby(
                    "TransactionType"
                )["IsFraud"]
                .mean()
                .mul(100)
                .reset_index(
                    name="FraudRate"
                )
            )

            fig = px.bar(
                type_data,
                x="TransactionType",
                y="FraudRate",
                title="Observed Fraud Rate by Transaction Type",
            )

            fig.update_traces(
                marker_color="#8A5CFF"
            )

            fig.update_yaxes(
                ticksuffix="%",
                title="Fraud Rate (%)",
            )

            st.plotly_chart(
                style_chart(fig),
                width="stretch",
            )

        # ----------------------------------------------------
        # DAY OF WEEK
        # ----------------------------------------------------

        elif intelligence_view == "Day of Week":

            day_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            day_data = (
                intelligence_df
                .groupby(
                    "DayOfWeek"
                )["IsFraud"]
                .mean()
                .mul(100)
                .reindex(
                    day_order
                )
                .reset_index(
                    name="FraudRate"
                )
            )

            fig = px.bar(
                day_data,
                x="DayOfWeek",
                y="FraudRate",
                title="Observed Fraud Rate by Day",
            )

            fig.update_traces(
                marker_color="#B7FF63"
            )

            fig.update_yaxes(
                ticksuffix="%",
                title="Fraud Rate (%)",
            )

            st.plotly_chart(
                style_chart(fig),
                width="stretch",
            )

        # ----------------------------------------------------
        # AMOUNT DISTRIBUTION
        # ----------------------------------------------------

        else:

            amount_data = (
                intelligence_df.copy()
            )

            amount_data["Status"] = (
                amount_data["IsFraud"]
                .map(
                    {
                        0: "Legitimate",
                        1: "Fraud",
                    }
                )
            )

            fig = px.box(
                amount_data,
                x="Status",
                y="Amount",
                color="Status",
                title="Transaction Amount Distribution",
            )

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                style_chart(fig),
                width="stretch",
            )

    # --------------------------------------------------------
    # KEY OBSERVATIONS
    # --------------------------------------------------------

    html_block(
        """
        <div class="section-header"
             style="margin-top:28px;">

            <div>

                <div style="
                    color:#F5F6FF;
                    font-size:24px;
                    font-weight:950;
                    letter-spacing:-0.025em;
                    line-height:1.1;
                ">
                    KEY OBSERVATIONS
                </div>

                <div style="
                    color:#68718F;
                    font-size:11px;
                    margin-top:5px;
                ">
                    Calculated from the current filter selection
                </div>

            </div>

        </div>
        """
    )

    if filtered_count == 0:

        st.info(
            "There are no observations available for "
            "the current filter selection."
        )

    else:

        observations = []

        # ----------------------------------------------------
        # OBSERVATION 1 — DATASET SIZE
        # ----------------------------------------------------

        observations.append(
            (
                "DATASET COVERAGE",
                f"{filtered_count:,} transactions "
                "match the current filters."
            )
        )

        # ----------------------------------------------------
        # OBSERVATION 2 — FRAUD RATE
        # ----------------------------------------------------

        observations.append(
            (
                "OBSERVED FRAUD RATE",
                (
                    f"The observed fraud rate in this "
                    f"selection is {filtered_rate:.2f}%."
                )
            )
        )

        # ----------------------------------------------------
        # OBSERVATION 3 — TRANSACTION AMOUNT
        # ----------------------------------------------------

        highest_amount = (
            intelligence_df[
                "Amount"
            ].max()
        )

        observations.append(
            (
                "TRANSACTION VALUE",
                (
                    f"The highest transaction amount "
                    f"in the current selection is "
                    f"${highest_amount:,.2f}."
                )
            )
        )

        # ----------------------------------------------------
        # OBSERVATION 4 — LOCATION
        # ----------------------------------------------------

        if "Location" in intelligence_df.columns:

            location_counts = (
                intelligence_df[
                    "Location"
                ]
                .value_counts()
            )

            if len(location_counts):

                observations.append(
                    (
                        "LOCATION COVERAGE",
                        (
                            f"{location_counts.index[0]} "
                            "is the most represented location "
                            "in the filtered dataset."
                        )
                    )
                )

        # ----------------------------------------------------
        # OBSERVATION 5 — TRANSACTION TYPE
        # ----------------------------------------------------

        if "TransactionType" in intelligence_df.columns:

            type_counts = (
                intelligence_df[
                    "TransactionType"
                ]
                .value_counts()
            )

            if len(type_counts):

                observations.append(
                    (
                        "TRANSACTION MIX",
                        (
                            f"{type_counts.index[0]} is the "
                            "most represented transaction type "
                            "in the filtered dataset."
                        )
                    )
                )

        # ----------------------------------------------------
        # OBSERVATION CARDS
        # ----------------------------------------------------

        observation_columns = st.columns(3)

        for index, observation in enumerate(
            observations
        ):

            with observation_columns[
                index % 3
            ]:

                html_block(
                    f"""
                    <div style="
                        background:
                            linear-gradient(
                                145deg,
                                rgba(255,255,255,0.045),
                                rgba(255,255,255,0.018)
                            );
                        border:
                            1px solid
                            rgba(138,92,255,0.18);
                        border-radius:16px;
                        padding:18px;
                        min-height:145px;
                        margin-bottom:14px;
                        box-shadow:
                            0 10px 30px
                            rgba(0,0,0,0.12);
                    ">

                        <div style="
                            color:#68718F;
                            font-size:10px;
                            font-weight:900;
                            letter-spacing:0.12em;
                            margin-bottom:10px;
                        ">
                            {observation[0]}
                        </div>

                        <div style="
                            color:#F5F6FF;
                            font-size:15px;
                            font-weight:850;
                            line-height:1.45;
                        ">
                            {observation[1]}
                        </div>

                    </div>
                    """
                )

        st.caption(
            "These observations describe patterns within "
            "the selected dataset. They should not be "
            "interpreted as proof that a location, merchant "
            "or transaction type is inherently suspicious."
        )
# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "DATA EXPLORER":

    show_hero(
        "DATA EXPLORER · TRANSACTION SEARCH",
        "Search the evidence.",
        (
            "Search and filter the reference transaction "
            "dataset without unnecessarily rendering the "
            "entire dataset."
        ),
        "DATA ACCESS READY",
    )


    section_header(
        "FILTER TRANSACTIONS",
        "Narrow the reference dataset",
    )


    e1, e2, e3 = st.columns(3)


    with e1:

        search_value = st.text_input(
            "Transaction / Merchant",
            placeholder="Enter ID...",
        )


    with e2:

        location_filter = st.selectbox(
            "Location",
            [
                "All"
            ]
            +
            sorted(
                df[
                    "Location"
                ]
                .dropna()
                .unique()
                .tolist()
            ),
        )


    with e3:

        status_filter = st.selectbox(
            "Status",
            [
                "All",
                "Legitimate",
                "Fraud",
            ],
        )


    filtered = df.copy()


    if search_value.strip():

        search_mask = pd.Series(
            False,
            index=filtered.index,
        )


        for column in [
            "TransactionID",
            "MerchantID",
        ]:

            if column in filtered.columns:

                search_mask = (
                    search_mask
                    |
                    filtered[
                        column
                    ]
                    .astype(str)
                    .str.contains(
                        search_value.strip(),
                        case=False,
                        na=False,
                    )
                )


        filtered = filtered[
            search_mask
        ]


    if location_filter != "All":

        filtered = filtered[
            filtered["Location"]
            ==
            location_filter
        ]


    if status_filter == "Fraud":

        filtered = filtered[
            filtered["IsFraud"]
            ==
            1
        ]

    elif status_filter == "Legitimate":

        filtered = filtered[
            filtered["IsFraud"]
            ==
            0
        ]


    metric_card(
        "MATCHING TRANSACTIONS",
        f"{len(filtered):,}",
        "Current search result",
        "accent-cyan",
    )


    explorer_columns = [
        column
        for column in [
            "TransactionID",
            "Amount",
            "MerchantID",
            "TransactionDate",
            "TransactionType",
            "Location",
            "IsFraud",
        ]
        if column in filtered.columns
    ]


    st.dataframe(
        filtered[
            explorer_columns
        ].head(500),
        width="stretch",
        height=540,
        hide_index=True,
    )


# ============================================================
# MODEL LAB
# ============================================================

elif page == "MODEL LAB":

    show_hero(
        "MODEL LAB · SYSTEM INTELLIGENCE",
        "Understand the engine.",
        (
            "Inspect the Random Forest model, the exact "
            "feature contract used for prediction and "
            "the health of the reference dataset."
        ),
        "MODEL SYSTEM READY",
    )


    section_header(
        "ENGINE STATUS",
        "Current FraudGuard configuration",
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        metric_card(
            "MODEL",
            "RANDOM FOREST",
            "Loaded model artifact",
            "accent-violet",
        )


    with m2:

        metric_card(
            "ROWS",
            f"{len(df):,}",
            "Reference dataset",
            "accent-cyan",
        )


    with m3:

        metric_card(
            "FEATURES",
            str(
                len(
                    MODEL_FEATURES
                )
            ),
            "Expected model inputs",
            "accent-lime",
        )


    with m4:

        metric_card(
            "MISSING",
            f"{int(df.isna().sum().sum()):,}",
            "Reference dataset",
            "accent-pink",
        )


    # --------------------------------------------------------
    # MODEL CONTRACT
    # --------------------------------------------------------

    section_header(
        "MODEL INPUT CONTRACT",
        "Features engineered before prediction",
    )


    feature_table = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES
        }
    )


    st.dataframe(
        feature_table,
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # DATA HEALTH
    # --------------------------------------------------------

    section_header(
        "DATA HEALTH",
        "Reference dataset integrity",
    )


    health_table = pd.DataFrame(
        {
            "Check": [
                "Rows",
                "Columns",
                "Duplicates",
                "Missing Cells",
                "Fraud Labels",
            ],

            "Value": [
                len(df),
                df.shape[1],

                int(
                    df.duplicated()
                    .sum()
                ),

                int(
                    df.isna()
                    .sum()
                    .sum()
                ),

                int(
                    df["IsFraud"]
                    .sum()
                ),
            ],
        }
    )


    st.dataframe(
        health_table,
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # MODEL EXPLANATION
    # --------------------------------------------------------

    section_header(
        "MODEL INTERPRETATION",
        "How FraudGuard should be understood",
    )


    html_block(
        """
        <div class="info-card">

            <div style="
                color:#39E7FF;
                font-size:10px;
                font-weight:900;
                letter-spacing:.08em;
                ">

                RANDOM FOREST · 15 FEATURE CONTRACT

            </div>

            <div style="
                color:#AEB7D5;
                font-size:9px;
                line-height:1.7;
                margin-top:10px;
                ">

                FraudGuard converts transaction information
                into the exact engineered feature set required
                by the trained Random Forest model.

                <br><br>

                The resulting probability is a model output.
                It should be treated as decision-support
                information rather than independent proof
                that a transaction is fraudulent.

            </div>

        </div>
        """
    )


    st.warning(
        "FraudGuard is a decision-support system. "
        "A model prediction is not, by itself, proof "
        "that a transaction is fraudulent."
    )


# ============================================================
# END OF FRAUDGUARD
# ============================================================
