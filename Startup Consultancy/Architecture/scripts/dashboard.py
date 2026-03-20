from __future__ import annotations

import math
import re
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

# ============================
# Page config & theme
# ============================
st.set_page_config(
    page_title="ComplianceIQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg-main: #061329;
        --bg-card: rgba(11, 25, 52, 0.92);
        --bg-card-2: rgba(15, 31, 61, 0.90);
        --line: rgba(110, 145, 200, 0.18);
        --text-main: #f4f7fb;
        --text-soft: #b8c7e1;
        --text-dim: #86a0c8;
        --accent: #59a5ff;
        --danger: #ff7a84;
        --ok: #59d38c;
        --warn: #ffca6b;
    }

    .stApp {
        background: radial-gradient(circle at top left, #07245d 0%, var(--bg-main) 45%, #041021 100%);
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 3.8rem !important;
        padding-bottom: 4rem !important;
        max-width: 1440px;
    }

    [data-testid="stSidebar"] {
        background: rgba(17, 20, 33, 0.92);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    html, body, [class*="css"]  {
        color: var(--text-main);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .page-title {
        font-size: 2.05rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0 0 0.65rem 0;
        color: var(--text-main);
    }

    .page-subtitle {
        font-size: 1.03rem;
        line-height: 1.8;
        color: var(--text-soft);
        margin: 0 0 1.15rem 0;
    }

    .hero-card,
    .info-card,
    .metric-card,
    .detail-card,
    .result-card,
    .note-card,
    .cluster-card {
        background: linear-gradient(180deg, rgba(13, 27, 54, 0.95) 0%, rgba(8, 19, 40, 0.93) 100%);
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.14);
    }

    .hero-card {
        padding: 1.9rem 2rem;
        min-height: 230px;
    }

    .hero-title {
        font-size: 2.55rem;
        line-height: 1.18;
        font-weight: 850;
        margin: 0 0 0.9rem 0;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.8;
        color: var(--text-soft);
        max-width: 760px;
        margin-bottom: 0;
    }

    .badge-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-content: flex-start;
        justify-content: flex-start;
        height: 100%;
        padding-top: 0.4rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.7rem 1.15rem;
        border-radius: 999px;
        border: 1px solid rgba(98, 166, 255, 0.35);
        background: rgba(31, 63, 112, 0.33);
        color: var(--text-main);
        font-weight: 700;
        font-size: 0.98rem;
        min-height: 48px;
    }

    .section-gap { margin-top: 1.6rem; }

    .hero-card.hero-main {
        background: linear-gradient(135deg, rgba(20, 40, 84, 0.95) 0%, rgba(9, 23, 48, 0.96) 100%);
    }

    .hero-card.hero-badges {
        background: linear-gradient(180deg, rgba(17, 34, 68, 0.94) 0%, rgba(10, 22, 46, 0.95) 100%);
    }

    .badge-pill.badge-a { background: rgba(66, 119, 255, 0.20); border-color: rgba(116, 165, 255, 0.36); }
    .badge-pill.badge-b { background: rgba(32, 131, 121, 0.22); border-color: rgba(81, 202, 187, 0.32); }
    .badge-pill.badge-c { background: rgba(96, 88, 182, 0.22); border-color: rgba(144, 136, 255, 0.30); }

    .home-hero-shell {
        background: linear-gradient(135deg, rgba(18, 42, 84, 0.96) 0%, rgba(8, 23, 46, 0.96) 100%);
        border: 1px solid rgba(104, 148, 221, 0.18);
        border-radius: 28px;
        padding: 2.2rem 2.35rem;
        box-shadow: 0 18px 40px rgba(0,0,0,0.18);
    }

    .home-hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(260px, 0.8fr);
        gap: 1.4rem;
        align-items: start;
    }

    .home-hero-title {
        font-size: 3.7rem;
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin: 0 0 1.02rem 0;
        color: #F2F7FF;
        max-width: 760px;
    }

    .hero-highlight-blue { color: #E8F3FF; }
    .hero-highlight-teal { color: #35D6C1; }
    .hero-highlight-amber { color: #FFB020; }

    .home-hero-subtitle {
        font-size: 1.45rem;
        line-height: 1.32;
        color: rgba(214, 228, 249, 0.97);
        margin: 0.1rem 0 1.02rem 0;
        max-width: 980px;
        font-weight: 780;
        letter-spacing: -0.01em;
    }

    .home-hero-supporting {
        font-size: 0.94rem;
        line-height: 1.62;
        color: rgba(162, 181, 212, 0.9);
        margin-top: 0;
        max-width: 760px;
        letter-spacing: 0.01em;
    }

    .home-hero-tagline {
        font-size: 0.86rem;
        line-height: 1.5;
        color: rgba(184, 199, 225, 0.82);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.6rem;
        font-weight: 800;
    }

    .home-badges-panel {
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
        justify-content: center;
        min-height: 100%;
        padding-left: 0.4rem;
    }

    .home-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: fit-content;
        min-height: 46px;
        padding: 0.65rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.96rem;
        border: 1px solid rgba(122, 163, 226, 0.24);
        color: var(--text-main);
    }

    .home-badge-a { background: rgba(47, 143, 138, 0.20); }
    .home-badge-b { background: rgba(58, 111, 162, 0.22); }
    .home-badge-c { background: rgba(78, 75, 160, 0.24); }

    .home-section-card {
        background: linear-gradient(180deg, rgba(11, 26, 53, 0.84) 0%, rgba(8, 20, 42, 0.82) 100%);
        border: 1px solid rgba(110, 145, 200, 0.12);
        border-radius: 24px;
        padding: 1.55rem 1.7rem;
    }

    .home-two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.4rem;
        align-items: stretch;
    }

    .home-col-title {
        font-size: 1.34rem;
        font-weight: 800;
        margin: 0 0 0.9rem 0;
        color: var(--text-main);
    }

    .home-col-accent {
        width: 44px;
        height: 3px;
        border-radius: 999px;
        margin: 0 0 0.95rem 0;
    }

    .home-col-accent-a { background: rgba(89, 165, 255, 0.86); }
    .home-col-accent-b { background: rgba(134, 118, 255, 0.86); }

    .home-bullet-list {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
        margin: 0;
    }

    .home-bullet-item {
        display: grid;
        grid-template-columns: 14px 1fr;
        gap: 0.7rem;
        align-items: start;
        color: var(--text-main);
        font-size: 0.99rem;
        line-height: 1.68;
    }

    .bullet-key-blue { color: #5AA9FF; font-weight: 750; }
    .bullet-key-teal { color: #35D6C1; font-weight: 750; }
    .bullet-key-amber { color: #FFB020; font-weight: 750; }

    .home-bullet-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        margin-top: 0.48rem;
    }

    .home-bullet-dot-a { background: rgba(89, 165, 255, 0.90); }
    .home-bullet-dot-b { background: rgba(134, 118, 255, 0.90); }

    .workflow-strip {
        border-radius: 20px;
        padding: 0.95rem 1.15rem;
        border: 1px solid rgba(110, 145, 200, 0.10);
        background: rgba(10, 21, 43, 0.56);
    }

    .workflow-text {
        text-align: center;
        color: #dfe9ff;
        letter-spacing: 0.01em;
        font-weight: 760;
        font-size: 1.18rem;
    }

    .workflow-arrow {
        color: rgba(73, 208, 187, 0.92);
        padding: 0 0.4rem;
        font-weight: 800;
    }

    .summary-shell {
        background: linear-gradient(180deg, rgba(10, 24, 49, 0.82) 0%, rgba(7, 18, 39, 0.80) 100%);
        border: 1px solid rgba(110, 145, 200, 0.10);
        border-radius: 24px;
        padding: 1.4rem 1.55rem 1.05rem 1.55rem;
    }

    .summary-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.15rem;
        margin-bottom: 0.95rem;
    }

    .summary-metric {
        padding: 0.45rem 0.25rem;
    }

    .summary-metric-label {
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-dim);
        font-size: 0.88rem;
        margin-bottom: 0.4rem;
    }

    .summary-metric-value {
        font-size: 2rem;
        font-weight: 820;
        color: var(--text-main);
        margin-bottom: 0.25rem;
        line-height: 1.15;
    }

    .metric-blue { color: #5AA9FF; }
    .metric-teal { color: #35D6C1; }
    .metric-amber { color: #FFB020; }

    .summary-metric-desc {
        color: var(--text-soft);
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .policy-note-inline {
        border-top: 1px solid rgba(110, 145, 200, 0.10);
        padding-top: 0.95rem;
        color: var(--text-dim);
        font-size: 0.88rem;
        line-height: 1.6;
        display: flex;
        gap: 0.55rem;
        align-items: flex-start;
    }

    .policy-note-icon {
        color: rgba(123, 168, 240, 0.85);
        font-size: 0.95rem;
        line-height: 1.1;
        margin-top: 0.05rem;
    }

    @media (max-width: 1200px) {
        .home-hero-grid { grid-template-columns: 1fr; }
        .home-badges-panel { padding-left: 0; flex-direction: row; flex-wrap: wrap; }
        .home-two-col { grid-template-columns: 1fr; }
        .summary-metrics { grid-template-columns: 1fr; }
    }

    .info-card {
        min-height: 205px;
    }

    .info-blue {
        background: linear-gradient(180deg, rgba(13, 34, 73, 0.96) 0%, rgba(9, 24, 50, 0.94) 100%);
    }

    .info-purple {
        background: linear-gradient(180deg, rgba(33, 24, 70, 0.96) 0%, rgba(11, 22, 48, 0.94) 100%);
    }

    .note-foot {
        min-height: auto;
        padding: 0.7rem 1rem;
        margin-top: 0.35rem;
        background: rgba(9, 18, 38, 0.45);
    }

    .note-foot-text {
        font-size: 0.88rem;
        line-height: 1.55;
        color: var(--text-dim);
    }

    .metric-soft {
        background: linear-gradient(180deg, rgba(12, 28, 60, 0.92) 0%, rgba(8, 21, 44, 0.93) 100%);
    }

    .chart-panel-compact {
        padding-bottom: 0.2rem;
    }

    .cluster-card-wrap {
        position: relative;
        margin-bottom: 1.2rem;
    }

    .cluster-action-wrap {
        margin-top: -0.15rem;
        margin-bottom: 1.15rem;
    }

    .cluster-action-wrap button {
        min-height: 42px;
        border-radius: 16px;
        border: 1px solid rgba(105, 156, 255, 0.22) !important;
        background: linear-gradient(180deg, rgba(17, 37, 74, 0.92) 0%, rgba(10, 24, 50, 0.95) 100%) !important;
        color: #dfe9ff !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
    }


    .info-card, .metric-card, .detail-card, .result-card, .note-card {
        padding: 1.35rem 1.45rem;
        height: 100%;
        min-height: 220px;
    }

    .info-card h3, .detail-card h3, .result-card h3 {
        font-size: 1.55rem;
        margin: 0 0 0.95rem 0;
        line-height: 1.3;
    }

    .info-card ul {
        margin: 0.25rem 0 0 0.8rem;
        padding-left: 1rem;
        color: var(--text-main);
        line-height: 1.8;
        font-size: 1rem;
    }

    .note-card {
        min-height: auto;
        padding: 1rem 1.35rem;
    }

    .note-text {
        color: var(--text-soft);
        line-height: 1.8;
        font-size: 1rem;
    }

    .metric-card {
        min-height: 150px;
    }

    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-dim);
        font-size: 0.95rem;
        margin-bottom: 0.65rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .metric-desc {
        color: var(--text-soft);
        line-height: 1.65;
        font-size: 1rem;
    }

    .subsection-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 1.4rem 0 0.95rem 0;
    }

    .cluster-card-wrap {
        position: relative;
        margin-bottom: 1.15rem;
    }

    .cluster-card {
        position: relative;
        min-height: 190px;
        padding: 1rem 1.25rem 0.95rem 1.25rem;
        overflow: hidden;
        margin-bottom: 0;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: linear-gradient(180deg, rgba(13, 29, 58, 0.96) 0%, rgba(8, 19, 40, 0.95) 100%);
    }

    .cluster-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 6px;
        background: var(--cluster-color, var(--accent));
        border-radius: 22px 0 0 22px;
    }

    .cluster-card.selected {
        border: 1px solid rgba(255,255,255,0.30);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.08), 0 18px 34px rgba(0,0,0,0.20);
        transform: translateY(-2px);
    }

    .cluster-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.22rem;
    }

    .cluster-type {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #9cb7e7;
        text-transform: uppercase;
        margin-bottom: 0;
    }

    .cluster-badge {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--cluster-color, var(--accent));
        box-shadow: 0 0 0 4px rgba(255,255,255,0.03);
        flex-shrink: 0;
    }

    .cluster-name {
        font-size: 1.62rem;
        font-weight: 860;
        line-height: 1.18;
        min-height: 2.45rem;
        margin-bottom: 0.12rem;
        color: var(--text-main);
    }

    .cluster-metrics {
        margin-top: 0;
    }

    .cluster-share {
        font-size: 2.05rem;
        font-weight: 900;
        line-height: 1.0;
        color: var(--cluster-color, var(--accent));
        margin-bottom: 0.08rem;
    }

    .cluster-count {
        color: var(--text-soft);
        font-size: 0.98rem;
        line-height: 1.35;
        margin-bottom: 0;
    }

    .detail-card {
        min-height: 230px;
    }

    .detail-card.tall {
        min-height: 280px;
    }

    .detail-card.compact {
        min-height: 165px;
    }

    .detail-card {
        border: 1px solid rgba(112, 149, 213, 0.14);
        box-shadow: 0 14px 28px rgba(0,0,0,0.14);
    }

    .detail-card.compact {
        background: linear-gradient(180deg, rgba(14, 31, 60, 0.94) 0%, rgba(8, 20, 42, 0.94) 100%);
    }

    .detail-card.tall {
        background: linear-gradient(180deg, rgba(13, 29, 56, 0.94) 0%, rgba(8, 20, 41, 0.94) 100%);
    }

    .detail-label {
        color: #8fb5eb;
    }

    .detail-value strong {
        color: #f6f8ff;
    }

    .detail-accent-blue {
        color: #7cb7ff;
        font-weight: 780;
    }

    .detail-accent-teal {
        color: #47dac7;
        font-weight: 760;
    }

    .detail-list-compact {
        line-height: 1.95;
    }

    .detail-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-dim);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .detail-value {
        font-size: 1rem;
        line-height: 1.8;
        color: var(--text-main);
        margin-bottom: 1rem;
    }
.detail-summary {
        color: #bdd0ef;
        font-size: 0.98rem;
        line-height: 1.72;
        margin: -0.15rem 0 1rem 0;
        max-width: 95%;
    }

    .detail-primary-title {
        font-size: 1.72rem;
        line-height: 1.24;
        font-weight: 850;
        color: #F5F9FF;
        margin: 0 0 0.45rem 0;
    }

    .detail-summary {
        color: #C6D8F2;
        font-size: 1rem;
        line-height: 1.8;
        margin: 0 0 1.05rem 0;
        max-width: 96%;
    }

    .detail-pill-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.85rem;
        margin: 0.35rem auto 0 auto;
        max-width: 440px;
    }

    .detail-two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        align-items: stretch;
    }

    .detail-col-block {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .detail-col-block .detail-list-grid {
        flex: 1;
    }

    .detail-risk-pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        margin: 0.15rem 0 0.8rem 0;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 0.82rem;
        font-weight: 760;
        letter-spacing: 0.02em;
    }

    .detail-item-line {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        width: 100%;
    }

    .detail-item-inline-meta {
        color: #9FC3F2;
        font-size: 0.95rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .cluster-risk-tag {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        margin-top: 0.18rem;
        padding: 0.24rem 0.62rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 760;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.06);
    }

    .decision-grid {
        display: grid;
        gap: 0.8rem;
        margin-top: 1rem;
    }

    .decision-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
    }

    .decision-outcome {
        font-size: 1.02rem;
        font-weight: 820;
        margin-bottom: 0.32rem;
    }

    .decision-label {
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8fb5eb;
        font-size: 0.78rem;
        margin-bottom: 0.18rem;
    }

    .decision-value {
        color: #f4f8ff;
        font-size: 0.96rem;
        line-height: 1.55;
    }

    .decision-green { color: #8ee2b2; }
    .decision-amber { color: #ffd37b; }
    .decision-orange { color: #ffb16e; }
    .decision-red { color: #ff8f97; }

    .detail-mini-pill {
        min-width: 180px;
        padding: 0.8rem 0.95rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(122, 163, 226, 0.14);
    }

    .detail-pill-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8FB5EB;
        font-size: 0.8rem;
        margin-bottom: 0.32rem;
    }

    .detail-pill-value {
        color: #F4F8FF;
        font-size: 1.03rem;
        font-weight: 760;
        line-height: 1.45;
    }

    .detail-card.primary-profile {
        min-height: 250px;
        padding: 1.45rem 1.55rem;
    }

    .detail-card.secondary-profile {
        min-height: 215px;
        padding: 1.35rem 1.45rem;
    }

    .detail-card.market-leaders {
        min-height: 485px;
        padding: 1.45rem 1.55rem;
    }

    .detail-section-label {
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #8FB5EB;
        font-size: 0.9rem;
        margin-bottom: 0.55rem;
    }

    .detail-list-grid {
        display: grid;
        gap: 0.8rem;
        margin-top: 0.15rem;
    }

    .detail-list-item {
        padding-bottom: 0.72rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .detail-list-item:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }

    .detail-rank {
        color: #5AA9FF;
        font-size: 0.96rem;
        font-weight: 780;
        margin-right: 0.4rem;
    }

    .detail-item-name {
        color: #F5F9FF;
        font-size: 1rem;
        font-weight: 720;
        line-height: 1.55;
    }

    .detail-item-meta {
        color: #9FC3F2;
        font-size: 0.95rem;
        font-weight: 680;
    }

    .detail-subgroup {
        margin-top: 0.15rem;
    }

    .detail-subgroup + .detail-subgroup {
        margin-top: 1.1rem;
    }

    .detail-subgroup-title {
        color: #CFE2FF;
        font-size: 1.02rem;
        font-weight: 760;
        margin-bottom: 0.7rem;
    }

    .detail-value .detail-emphasis {
        color: #DDEBFF;
        font-weight: 720;
    }

    .detail-strong-title {
        color: #ffffff;
        font-weight: 820;
        font-size: 1.24rem;
        line-height: 1.4;
        margin-bottom: 0.35rem;
    }

    .detail-emphasis {
        color: #5AA9FF;
        font-weight: 780;
    }

    .chart-panel {
        margin-bottom: 1rem;
    }

    .chart-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.35rem 0 0.65rem 0.2rem;
    }

    .result-card {
        min-height: 175px;
    }

    .result-kicker {
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-dim);
        font-size: 0.95rem;
        margin-bottom: 0.7rem;
    }

    .result-main {
        font-size: 1.45rem;
        line-height: 1.4;
        font-weight: 850;
        margin-bottom: 0.5rem;
    }

    .result-sub {
        color: var(--text-soft);
        line-height: 1.65;
        font-size: 0.98rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.58rem 1rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 800;
        border: 1px solid rgba(255,255,255,0.18);
        margin-bottom: 0.5rem;
    }

    .status-ok {
        background: rgba(35, 103, 62, 0.24);
        color: #aef0c7;
        border-color: rgba(89, 211, 140, 0.32);
    }

    .status-warn {
        background: rgba(111, 88, 34, 0.22);
        color: #ffe7a7;
        border-color: rgba(255, 202, 107, 0.28);
    }

    .status-bad {
        background: rgba(116, 38, 45, 0.22);
        color: #ffc1c7;
        border-color: rgba(255, 122, 132, 0.28);
    }

    .sidebar-mini {
        color: var(--text-soft);
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .small-muted {
        color: var(--text-dim);
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .stButton button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(120, 148, 193, 0.22);
        background: rgba(255,255,255,0.03);
        color: #f6f8ff;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton button:hover {
        border-color: rgba(142, 176, 230, 0.4);
        background: rgba(255,255,255,0.05);
        color: white;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(8, 18, 39, 0.42);
    }

    .brand-kpi-card {
        background: linear-gradient(180deg, rgba(13, 30, 58, 0.94) 0%, rgba(8, 20, 42, 0.94) 100%);
        border: 1px solid rgba(110, 145, 200, 0.12);
        border-radius: 22px;
        padding: 1.2rem 1.35rem;
        min-height: 150px;
    }

    .brand-kpi-label {
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-dim);
        font-size: 0.9rem;
        margin-bottom: 0.45rem;
    }

    .brand-kpi-value {
        font-size: 2rem;
        line-height: 1.08;
        font-weight: 850;
        margin-bottom: 0.35rem;
        color: #f4f8ff;
    }

    .brand-kpi-value.accent-red { color: #ff7a84; }
    .brand-kpi-value.accent-amber { color: #ffca6b; }
    .brand-kpi-value.accent-blue { color: #7db8ff; }

    .brand-kpi-desc {
        color: var(--text-soft);
        font-size: 0.96rem;
        line-height: 1.55;
    }

    .brand-detail-shell {
        background: linear-gradient(180deg, rgba(13, 29, 56, 0.95) 0%, rgba(8, 20, 41, 0.95) 100%);
        border: 1px solid rgba(110, 145, 200, 0.14);
        border-radius: 24px;
        padding: 1.15rem 1.25rem;
        min-height: auto;
        margin-bottom: 0.95rem;
    }

    .brand-detail-title {
        font-size: 1.75rem;
        font-weight: 860;
        line-height: 1.18;
        color: #f6f9ff;
        margin-bottom: 0.55rem;
    }

    .brand-detail-sub {
        color: #c8d8f1;
        font-size: 1rem;
        line-height: 1.75;
        margin-bottom: 1rem;
    }

    .brand-detail-shell.profile-main {
        padding: 1.1rem 1.2rem;
    }

    .brand-detail-shell.profile-main .brand-detail-sub {
        margin-bottom: 0.7rem;
    }

    .brand-detail-shell.profile-main .brand-pill-row {
        margin: 0.2rem 0 0 0;
    }

    .brand-detail-shell.profile-mini .brand-grid-two {
        margin-bottom: 0;
        gap: 0.8rem;
    }

    .brand-detail-shell.profile-action .brand-action-text,
    .brand-detail-shell.profile-action .brand-issue-text {
        margin-bottom: 0.7rem;
    }

    .brand-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin: 0.35rem 0 1rem 0;
    }

    .brand-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.62rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(122, 163, 226, 0.14);
        color: #eef5ff;
        font-size: 0.95rem;
        line-height: 1.35;
    }

    .brand-pill .meta {
        color: #91b4e8;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .brand-grid-two {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.95rem;
        margin-bottom: 1rem;
    }

    .brand-mini-card {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(122, 163, 226, 0.12);
    }

    .brand-mini-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8fb5eb;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }

    .brand-mini-value {
        color: #f6f9ff;
        font-size: 1.12rem;
        font-weight: 770;
        line-height: 1.45;
    }

    .brand-section-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8fb5eb;
        font-size: 0.86rem;
        margin-bottom: 0.45rem;
    }

    .brand-section-title {
        color: #dce9ff;
        font-size: 1.02rem;
        font-weight: 770;
        margin: 0.15rem 0 0.45rem 0;
    }

    .brand-issue-text, .brand-action-text {
        color: #d7e4f8;
        line-height: 1.78;
        font-size: 0.98rem;
        margin-bottom: 0.95rem;
    }

    .brand-mix-list {
        display: grid;
        gap: 0.55rem;
    }

    .brand-mix-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.85rem;
        color: #eef4ff;
        font-size: 0.98rem;
    }

    .brand-mix-item .mix-share {
        color: #8ec7ff;
        font-weight: 760;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Constants
# ============================
DEFAULT_TARGET_SCHEDULE = {
    2012: 174, 2013: 169, 2014: 164, 2015: 157, 2016: 150,
    2017: 142, 2018: 135, 2019: 128, 2020: 122, 2021: 119,
    2022: 117, 2023: 107, 2024: 102, 2025: 96, 2026: 86,
}

CLUSTER_ORDER = [
    "EV Drivers",
    "Large SUVs",
    "Hybrid Transition",
    "Urban Commuters",
    "Family Cars",
    "Work Trucks",
    "Practical SUVs",
    "Performance Cars",
]

CLUSTER_COLORS = {
    "EV Drivers": "#3b82f6",
    "Large SUVs": "#ff5c5c",
    "Hybrid Transition": "#8b5cf6",
    "Urban Commuters": "#1dd3b0",
    "Family Cars": "#f59e0b",
    "Work Trucks": "#f97316",
    "Practical SUVs": "#2dd4bf",
    "Performance Cars": "#ec4899",
}

CLUSTER_RISK_CATEGORY = {
    "Work Trucks": "Critical Risk",
    "Performance Cars": "Critical Risk",
    "Family Cars": "High Risk",
    "Large SUVs": "High Risk",
    "Urban Commuters": "Moderate Risk",
    "Practical SUVs": "Moderate Risk",
    "EV Drivers": "Zero / Low Risk",
    "Hybrid Transition": "Zero / Low Risk",
}

CLUSTER_RISK_COLOR = {
    "Critical Risk": "#ff8a94",
    "High Risk": "#ffb020",
    "Moderate Risk": "#f2d27a",
    "Zero / Low Risk": "#7ee1ac",
}

FIELD_CANDIDATES = {
    "cluster": ["cluster_name", "cluster", "cluster_label", "cluster_desc"],
    "co2": ["CO2_Emissions_gkm", "co2_emissions_gkm", "CO2_Emissions", "co2_emissions", "co2", "CO2"],
    "engine": ["Engine_Size_L", "engine_size_l", "engine_size", "Engine_Size"],
    "energy": ["Energy_Consumption_MJ100km", "energy_consumption_mj_100km", "energy_consumption", "Energy_Consumption", "Fuel_Cons_Comb_MPKM", "Fuel_Cons_Comb_MJ100km"],
    "fuel_cons": ["Fuel_Cons_Comb_L100km", "fuel_cons_comb_l100km", "fuel_consumption_combined", "fuel_cons_comb"],
    "cylinders": ["Cylinders", "cylinders"],
    "fuel_group": ["fuel_group", "Fuel_Group", "fuel_type", "Fuel_Type"],
    "size_class": ["size_class", "Size_Class", "vehicle_class", "Vehicle_Class"],
    "trans_group": ["trans_group", "Trans_Group", "transmission_group"],
    "phev_flag": ["phev_flag", "PHEV_Flag", "is_phev"],
    "model_year": ["Model_Year", "model_year", "year"],
    "make": ["Make", "make", "manufacturer"],
    "model": ["Model", "model"],
    "powertrain": ["powertrain_group", "Powertrain_Group", "powertrain", "Powertrain"],
    "vehicle_class": ["Vehicle_Class", "vehicle_class", "size_class", "Size_Class"],
}

DISPLAY_LABELS = {
    "size_class": "Size class",
    "trans_group": "Transmission group",
    "fuel_group": "Fuel group",
    "engine": "Engine size (L)",
    "cylinders": "Cylinders",
    "phev_flag": "PHEV flag",
    "fuel_cons": "Fuel consumption combined (L/100 km)",
    "model_year": "Model year",
}


def pick_dashboard_file(filename: str) -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "dashboard" / filename,
        here.parent / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_target_schedule_from_workbook() -> Dict[int, float]:
    path = pick_dashboard_file("co2_target_values_by_model_year.xlsx")
    if path is None:
        return DEFAULT_TARGET_SCHEDULE.copy()
    try:
        raw = pd.read_excel(path, sheet_name="CO2_Targets", header=None, engine="openpyxl")
        data = raw.iloc[2:, :2].copy()
        data.columns = ["model_year", "target"]
        data["model_year"] = pd.to_numeric(data["model_year"], errors="coerce")
        data["target"] = pd.to_numeric(data["target"], errors="coerce")
        data = data.dropna()
        if data.empty:
            return DEFAULT_TARGET_SCHEDULE.copy()
        return {int(r.model_year): float(r.target) for r in data.itertuples(index=False)}
    except Exception:
        return DEFAULT_TARGET_SCHEDULE.copy()


def load_market_package() -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[Path]]:
    path = pick_dashboard_file("market_overview_data.xlsx")
    if path is None:
        return None, None
    try:
        overview = pd.read_excel(path, sheet_name="cluster_overview", engine="openpyxl")
        boxstats = pd.read_excel(path, sheet_name="cluster_boxplot_stats", engine="openpyxl")
        market_share = pd.read_excel(path, sheet_name="cluster_market_share", engine="openpyxl")

        summary = overview.rename(columns={
            "cluster_name": "cluster",
            "vehicle_type_category": "dominant_powertrain",
            "most_common_fuel_type": "common_fuel",
            "top_vehicle_classes": "top_classes",
            "top_5_makes_by_count": "top_makes",
            "top_5_models_by_count": "top_models",
            "description": "description",
            "share_of_full_market_pct": "share",
            "record_count": "count",
            "co2_mean_g_km": "avg_co2",
            "energy_consumption_mj_100km": "avg_energy",
        }).copy()
        summary["share"] = pd.to_numeric(summary["share"], errors="coerce")
        if summary["share"].dropna().max() <= 1.0:
            summary["share"] = summary["share"] * 100.0
        summary["count"] = pd.to_numeric(summary["count"], errors="coerce")
        summary["avg_co2"] = pd.to_numeric(summary["avg_co2"], errors="coerce")
        summary["avg_energy"] = pd.to_numeric(summary["avg_energy"], errors="coerce")
        summary = summary[[c for c in [
            "cluster", "count", "share", "dominant_powertrain", "description", "common_fuel",
            "top_classes", "top_makes", "top_models", "avg_co2", "avg_energy",
            "emissions_risk_label", "engine_size_mean_l"
        ] if c in summary.columns]]

        market_share = market_share.rename(columns={
            "cluster_name": "cluster",
            "share_of_full_market_pct": "share",
            "record_count": "count",
        }).copy()
        market_share["share"] = pd.to_numeric(market_share["share"], errors="coerce")
        if market_share["share"].dropna().max() <= 1.0:
            market_share["share"] = market_share["share"] * 100.0
        market_share["count"] = pd.to_numeric(market_share["count"], errors="coerce")

        boxstats = boxstats.rename(columns={"cluster_name": "cluster"}).copy()
        for c in boxstats.columns:
            if c != "cluster" and c != "engine_size_note":
                boxstats[c] = pd.to_numeric(boxstats[c], errors="coerce")

        return {
            "summary": summary,
            "market_share": market_share,
            "boxstats": boxstats,
        }, path
    except Exception:
        return None, path

# ============================
# Helpers
# ============================
def find_column(df: pd.DataFrame, key: str) -> Optional[str]:
    candidates = FIELD_CANDIDATES.get(key, [])
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def pick_existing_path(custom_path: Optional[str]) -> Optional[Path]:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "gold" / "vehicles_gold_ml.csv",
        here.parent.parent / "data" / "gold" / "vehicles_gold.csv",
        here.parent.parent / "data" / "gold" / "vehicle_type_summary.csv",
        here.parent / "vehicles_gold_ml.csv",
        Path.cwd() / "vehicles_gold_ml.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def infer_cluster_name(row: pd.Series) -> str:
    label = str(row.get("cluster", "")).strip()
    if label in CLUSTER_ORDER:
        return label

    powertrain = str(row.get("powertrain", "Unknown")).strip().upper()
    co2 = float(row.get("co2", np.nan)) if pd.notna(row.get("co2", np.nan)) else np.nan
    engine = float(row.get("engine", np.nan)) if pd.notna(row.get("engine", np.nan)) else np.nan
    fuel = str(row.get("fuel_group", "")).lower()
    size = str(row.get("size_class", "")).lower()
    model = str(row.get("model", "")).lower()

    if powertrain == "BEV" or co2 == 0:
        return "EV Drivers"
    if powertrain == "PHEV":
        return "Hybrid Transition"
    if any(x in size for x in ["compact", "subcompact", "minicompact"]) and (pd.isna(engine) or engine <= 2.0):
        return "Urban Commuters"
    if any(x in size for x in ["pickup", "van", "special purpose"]) or (pd.notna(engine) and engine >= 5.2):
        return "Work Trucks"
    if any(x in size for x in ["suv", "utility"]) and pd.notna(engine) and engine >= 3.8 and pd.notna(co2) and co2 >= 260:
        return "Large SUVs"
    if any(x in size for x in ["suv", "utility"]) and pd.notna(co2) and co2 <= 240:
        return "Practical SUVs"
    if any(x in size for x in ["two-seater", "full-size", "mid-size"]) and (pd.notna(engine) and engine >= 3.5 or any(k in model for k in ["amg", "m", "rs", "turbo"])):
        return "Performance Cars"
    return "Family Cars"


def normalise_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    mapping = {key: find_column(df, key) for key in FIELD_CANDIDATES}
    work = df.copy()

    renamed = {}
    for key, col in mapping.items():
        if col:
            renamed[col] = key
    work = work.rename(columns=renamed)

    for col in ["co2", "engine", "energy", "fuel_cons", "cylinders", "model_year", "phev_flag"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "cluster" not in work.columns:
        work["cluster"] = work.apply(infer_cluster_name, axis=1)
    else:
        work["cluster"] = work.apply(infer_cluster_name, axis=1)

    if "powertrain" not in work.columns:
        def infer_powertrain(r: pd.Series) -> str:
            fuel = str(r.get("fuel_group", "")).lower()
            if r.get("phev_flag", 0) == 1:
                return "PHEV"
            if "electric" in fuel or (pd.notna(r.get("co2", np.nan)) and float(r.get("co2", 999)) == 0):
                return "BEV"
            return "ICE & HEV"
        work["powertrain"] = work.apply(infer_powertrain, axis=1)

    return work, mapping


def load_data(custom_path: Optional[str]) -> Tuple[pd.DataFrame, Optional[Path], Dict[str, Optional[str]]]:
    path = pick_existing_path(custom_path)
    if path is None:
        raise FileNotFoundError("Could not find a vehicle dataset. Please provide a valid CSV path in the sidebar.")
    df = pd.read_csv(path)
    df, mapping = normalise_dataframe(df)
    return df, path, mapping



def load_model() -> Optional[object]:
    here = Path(__file__).resolve().parent
    model_path = here / "best_co2_model.pkl"
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def top_counts(series: pd.Series, top_n: int = 5) -> str:
    vals = series.dropna().astype(str)
    if vals.empty:
        return "Not available"
    counts = vals.value_counts().head(top_n)
    return ", ".join([f"{idx} ({val})" for idx, val in counts.items()])


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for cluster in CLUSTER_ORDER:
        sub = df[df["cluster"] == cluster].copy()
        if sub.empty:
            continue
        share = len(sub) / total * 100 if total else 0
        dominant_powertrain = sub["powertrain"].astype(str).value_counts().index[0] if "powertrain" in sub.columns else "Unknown"
        common_fuel = sub["fuel_group"].astype(str).replace("nan", np.nan).dropna().value_counts().index[0] if "fuel_group" in sub.columns and sub["fuel_group"].dropna().shape[0] else "Unknown"
        rows.append({
            "cluster": cluster,
            "count": len(sub),
            "share": share,
            "dominant_powertrain": dominant_powertrain,
            "common_fuel": common_fuel,
            "top_classes": top_counts(sub["vehicle_class"] if "vehicle_class" in sub.columns else pd.Series(dtype=object)),
            "top_makes": top_counts(sub["make"] if "make" in sub.columns else pd.Series(dtype=object)),
            "top_models": top_counts(sub["model"] if "model" in sub.columns else pd.Series(dtype=object)),
            "avg_co2": float(sub["co2"].mean()) if "co2" in sub.columns else np.nan,
            "avg_energy": float(sub["energy"].mean()) if "energy" in sub.columns and sub["energy"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def compute_box_stats(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for cluster in CLUSTER_ORDER:
        sub = df.loc[df["cluster"] == cluster, value_col].dropna().astype(float)
        if sub.empty:
            continue
        q1 = float(np.percentile(sub, 25))
        med = float(np.percentile(sub, 50))
        q3 = float(np.percentile(sub, 75))
        rows.append({
            "cluster": cluster,
            "min": float(sub.min()),
            "q1": q1,
            "median": med,
            "q3": q3,
            "max": float(sub.max()),
            "mean": float(sub.mean()),
        })
    return pd.DataFrame(rows)


def make_boxplot(df: pd.DataFrame, value_col: str, title: str, y_title: str, selected_cluster: Optional[str] = None) -> go.Figure:
    stats = compute_box_stats(df, value_col)
    fig = go.Figure()
    for _, r in stats.iterrows():
        color = CLUSTER_COLORS.get(r["cluster"], "#59a5ff")
        line_width = 3 if r["cluster"] == selected_cluster else 2
        fill_rgba = "rgba(255,255,255,0.02)" if r["cluster"] != selected_cluster else "rgba(255,255,255,0.08)"
        fig.add_trace(go.Box(
            x=[r["cluster"]],
            name=r["cluster"],
            q1=[r["q1"]],
            median=[r["median"]],
            q3=[r["q3"]],
            lowerfence=[max(0, r["min"])] if value_col == "co2" else [r["min"]],
            upperfence=[r["max"]],
            boxpoints=False,
            line=dict(color=color, width=line_width),
            fillcolor=fill_rgba,
            hovertemplate=(
                f"<b>{r['cluster']}</b><br>"
                f"Min: {r['min']:.1f}<br>"
                f"Q1: {r['q1']:.1f}<br>"
                f"Median: {r['median']:.1f}<br>"
                f"Q3: {r['q3']:.1f}<br>"
                f"Max: {r['max']:.1f}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=28, color="#f4f7fb")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        font=dict(color="#f4f7fb", size=14),
        margin=dict(l=20, r=20, t=82, b=40),
        height=500,
        showlegend=False,
        xaxis=dict(title="", tickangle=-28, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.08)", rangemode="tozero" if value_col == "co2" else "normal"),
    )
    if value_col == "co2":
        fig.update_yaxes(range=[0, max(1.0, float(stats["max"].max()) * 1.08)] if not stats.empty else [0, 1])
    return fig




def make_boxplot_from_precomputed(
    stats: pd.DataFrame,
    prefix: str,
    title: str,
    y_title: str,
    selected_cluster: Optional[str] = None,
    note: Optional[str] = None,
) -> go.Figure:
    cols = {
        "min": f"{prefix}_min_l" if prefix == "engine_size" else f"{prefix}_min_g_km",
        "q1": f"{prefix}_q1_l" if prefix == "engine_size" else f"{prefix}_q1_g_km",
        "median": f"{prefix}_median_l" if prefix == "engine_size" else f"{prefix}_median_g_km",
        "q3": f"{prefix}_q3_l" if prefix == "engine_size" else f"{prefix}_q3_g_km",
        "max": f"{prefix}_max_l" if prefix == "engine_size" else f"{prefix}_max_g_km",
    }
    work = stats.copy()
    fig = go.Figure()
    for _, r in work.iterrows():
        cluster = r["cluster"]
        color = CLUSTER_COLORS.get(cluster, "#59a5ff")
        line_width = 3 if cluster == selected_cluster else 2
        fill_rgba = "rgba(255,255,255,0.02)" if cluster != selected_cluster else "rgba(255,255,255,0.08)"
        fig.add_trace(go.Box(
            x=[cluster],
            name=cluster,
            q1=[r[cols["q1"]]],
            median=[r[cols["median"]]],
            q3=[r[cols["q3"]]],
            lowerfence=[r[cols["min"]]],
            upperfence=[r[cols["max"]]],
            boxpoints=False,
            line=dict(color=color, width=line_width),
            fillcolor=fill_rgba,
            hovertemplate=(
                f"<b>{cluster}</b><br>"
                f"Min: {r[cols['min']]:.1f}<br>"
                f"Q1: {r[cols['q1']]:.1f}<br>"
                f"Median: {r[cols['median']]:.1f}<br>"
                f"Q3: {r[cols['q3']]:.1f}<br>"
                f"Max: {r[cols['max']]:.1f}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=28, color="#f4f7fb")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        font=dict(color="#f4f7fb", size=14),
        margin=dict(l=20, r=20, t=82, b=40),
        height=500,
        showlegend=False,
        xaxis=dict(title="", tickangle=-28, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig

def make_donut(summary: pd.DataFrame, selected_cluster: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    work = summary.copy()
    work["share"] = pd.to_numeric(work.get("share"), errors="coerce")
    work["count"] = pd.to_numeric(work.get("count"), errors="coerce")
    hover_share = work["share"].fillna(0.0).tolist()
    hover_count = work["count"].fillna(0).astype(int).tolist()
    hover_text = [
        f"<b>{cluster}</b><br>Market share: {share:.2f}%<br>Vehicle count: {count:,}<extra></extra>"
        for cluster, share, count in zip(work["cluster"], hover_share, hover_count)
    ]

    fig = go.Figure(go.Pie(
        labels=work["cluster"],
        values=work["count"],
        hole=0.62,
        sort=False,
        marker=dict(colors=[CLUSTER_COLORS.get(c, "#59a5ff") for c in work["cluster"]], line=dict(color="#071221", width=2)),
        hovertext=hover_text,
        hovertemplate="%{hovertext}",
        textinfo="percent",
        textposition="inside",
        insidetextfont=dict(size=15, color="white"),
        pull=[0.05 if c == selected_cluster else 0 for c in work["cluster"]],
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f4f7fb", size=14),
        height=590,
        margin=dict(l=10, r=10, t=8, b=125),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, font=dict(size=11), itemwidth=70),
        annotations=[dict(text=f"<b>{len(work)}</b><br>clusters", x=0.5, y=0.5, font=dict(size=20, color="#f4f7fb"), showarrow=False)],
    )
    return fig

def make_bar(summary: pd.DataFrame, y_col: str, title: str, y_title: str, selected_cluster: Optional[str] = None) -> go.Figure:
    work = summary.dropna(subset=[y_col]).copy()
    fig = go.Figure(go.Bar(
        x=work["cluster"],
        y=work[y_col],
        marker=dict(color=[CLUSTER_COLORS.get(c, "#59a5ff") for c in work["cluster"]], line=dict(width=[3 if c == selected_cluster else 0 for c in work["cluster"]], color="#f5f7ff")),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=28, color="#f4f7fb")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        font=dict(color="#f4f7fb", size=14),
        margin=dict(l=20, r=20, t=60, b=50),
        height=420,
        xaxis=dict(title="", tickangle=-28),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.08)", rangemode="tozero"),
        showlegend=False,
    )
    return fig


def fallback_predict(inputs: Dict[str, float]) -> float:
    base = 12.0
    fuel_cons = float(inputs.get("fuel_cons", 0) or 0)
    engine = float(inputs.get("engine", 0) or 0)
    cylinders = float(inputs.get("cylinders", 0) or 0)
    phev = float(inputs.get("phev_flag", 0) or 0)
    fuel_group = str(inputs.get("fuel_group", "Unknown")).lower()
    trans_group = str(inputs.get("trans_group", "")).upper()
    size_class = str(inputs.get("size_class", "")).lower()

    co2 = base + fuel_cons * 20.2 + engine * 10.5 + cylinders * 3.2
    if phev == 1:
        co2 *= 0.62
    if "electric" in fuel_group:
        co2 = 0
    elif "diesel" in fuel_group:
        co2 *= 1.05
    elif "premium" in fuel_group:
        co2 *= 1.03

    if trans_group in {"A8", "A9", "A10"}:
        co2 *= 0.98
    elif trans_group in {"M5", "M6"}:
        co2 *= 1.02

    if any(k in size_class for k in ["compact", "subcompact", "minicompact"]):
        co2 *= 0.95
    elif any(k in size_class for k in ["pickup", "van", "special purpose"]):
        co2 *= 1.10
    elif "suv" in size_class:
        co2 *= 1.05

    return max(0.0, round(co2, 1))




def map_fuel_primary(code: object) -> str:
    if pd.isna(code):
        return "fossil"
    code = str(code).strip()
    if code in ("X", "Z", "D"):
        return "fossil"
    if code in ("E", "N"):
        return "alt_fuel"
    return "phev_elec"

def map_vehicle_class_raw(vc: object) -> str:
    small = {"minicompact", "subcompact", "compact", "two-seater", "station wagon - small", "station wagon: small"}
    mid = {"mid-size", "mid_size", "full-size", "full_size", "station wagon - mid-size", "station wagon: mid-size"}
    suvt = {"suv - small", "suv - standard", "sport utility vehicle: small", "sport utility vehicle: standard",
            "pickup truck - small", "pickup truck - standard", "pickup truck: standard", "special purpose vehicle"}
    vans = {"minivan", "van - passenger", "van - cargo"}
    if pd.isna(vc):
        return "other"
    v = str(vc).strip().lower()
    if v in small:
        return "small"
    if v in mid:
        return "mid"
    if v in suvt:
        return "suv_truck"
    if v in vans:
        return "van"
    return "other"

def map_transmission_raw(t: object) -> str:
    if pd.isna(t):
        return "unknown"
    t = str(t).strip().upper()
    if t.startswith("AM"):
        return "automated_manual"
    if t.startswith("AS") or t.startswith("A"):
        return "automatic"
    if t.startswith("AV"):
        return "cvt"
    if t.startswith("M"):
        return "manual"
    return "other"

def _artifact_transform_row(artifact: Dict[str, object], raw_inputs: Dict[str, object]) -> np.ndarray:
    engineered = {
        "Engine_Size_L": float(raw_inputs.get("Engine_Size_L", np.nan) or np.nan),
        "Cylinders": float(raw_inputs.get("Cylinders", np.nan) or np.nan),
        "fuel_group": map_fuel_primary(raw_inputs.get("Fuel_Type_Primary", np.nan)),
        "size_class": map_vehicle_class_raw(raw_inputs.get("Vehicle_Class", np.nan)),
        "trans_group": map_transmission_raw(raw_inputs.get("Transmission", np.nan)),
        "phev_flag": int(raw_inputs.get("is_phev", 0) or 0),
        "Fuel_Cons_Comb_L100km": float(raw_inputs.get("Fuel_Cons_Comb_L100km", np.nan) or np.nan),
    }
    feature_cols = artifact["feature_cols"]
    sub = pd.DataFrame([engineered])[feature_cols].copy()
    cat_cols = [c for c in feature_cols if sub[c].apply(lambda v: isinstance(v, str)).any()]
    if cat_cols:
        sub = pd.get_dummies(sub, columns=cat_cols, drop_first=True, dtype=float)
    for col in artifact["ohe_columns"]:
        if col not in sub.columns:
            sub[col] = 0.0
    sub = sub[artifact["ohe_columns"]].astype(float)
    x = sub.values.astype(float)
    mean = np.asarray(artifact["scaler_mean"], dtype=float)
    scale = np.asarray(artifact["scaler_scale"], dtype=float)
    x_scaled = (x - mean) / scale
    return x_scaled

def model_predict(model: object, inputs: Dict[str, object], columns: List[str]) -> float:
    if model is None:
        return fallback_predict({
            "fuel_cons": inputs.get("Fuel_Cons_Comb_L100km", 0),
            "engine": inputs.get("Engine_Size_L", 0),
            "cylinders": inputs.get("Cylinders", 0),
            "phev_flag": inputs.get("is_phev", 0),
            "fuel_group": inputs.get("Fuel_Type_Primary", ""),
            "size_class": inputs.get("Vehicle_Class", ""),
            "trans_group": inputs.get("Transmission", ""),
        })
    try:
        if isinstance(model, dict) and model.get("artifact_type") == "co2_linear_pipeline_artifact":
            x_scaled = _artifact_transform_row(model, inputs)
            coef = np.asarray(model["model_coef"], dtype=float)
            intercept = float(model["model_intercept"])
            pred = float((x_scaled @ coef.reshape(-1, 1)).ravel()[0] + intercept)
            return max(0.0, round(pred, 1))
        row = pd.DataFrame([{c: inputs.get(c, np.nan) for c in columns}])
        pred = model.predict(row)[0]
        return max(0.0, round(float(pred), 1))
    except Exception:
        return fallback_predict({
            "fuel_cons": inputs.get("Fuel_Cons_Comb_L100km", 0),
            "engine": inputs.get("Engine_Size_L", 0),
            "cylinders": inputs.get("Cylinders", 0),
            "phev_flag": inputs.get("is_phev", 0),
            "fuel_group": inputs.get("Fuel_Type_Primary", ""),
            "size_class": inputs.get("Vehicle_Class", ""),
            "trans_group": inputs.get("Transmission", ""),
        })

def get_model_features(model: object) -> List[str]:
    if model is None:
        return []
    if isinstance(model, dict) and model.get("artifact_type") == "co2_linear_pipeline_artifact":
        return list(model.get("raw_input_cols", []))
    for attr in ["feature_names_in_", "feature_name_", "feature_names"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (list, tuple, np.ndarray)):
                return [str(v) for v in val]
    return ["Engine_Size_L", "Cylinders", "Fuel_Type_Primary", "Vehicle_Class", "Transmission", "is_phev", "Fuel_Cons_Comb_L100km"]

def recommendation_text(pred: float, target: float) -> str:
    margin = target - pred
    if margin >= 15:
        return "Strong emissions profile for the selected model year."
    if margin >= 0:
        return "Compliant, but with limited margin against the target."
    return "Not compliant for the selected model year; further efficiency or electrification improvements are recommended."


def compliance_status(pred: float, target: float) -> Tuple[str, str]:
    margin = target - pred
    if pred <= target:
        if margin >= 15:
            return "Compliant", "status-ok"
        return "Compliant", "status-warn"
    return "Not Compliant", "status-bad"


def get_cluster_risk(cluster_name: str) -> tuple[str, str]:
    category = CLUSTER_RISK_CATEGORY.get(str(cluster_name), "Moderate Risk")
    color = CLUSTER_RISK_COLOR.get(category, "#f2d27a")
    return category, color

def classify_emissions_outcome(pred: float) -> dict:
    if pred <= 54.4:
        return {
            "outcome": "Fully compliant",
            "class": "decision-green",
            "band": "≤ 54.4 g/km",
            "horizon": "Compliant for 5+ years",
            "decision": "Approve",
            "action": "Fast-track approval. No further action is required. Log in compliance register as long-term safe.",
        }
    if pred <= 85.5:
        return {
            "outcome": "Compliant now",
            "class": "decision-amber",
            "band": "54.5–85.5 g/km",
            "horizon": "Compliant for approx. 3–5 years",
            "decision": "Approve with note",
            "action": "Approve for current cycle. Notify manufacturer that this model will fall out of compliance by 2028–2031. Require an electrification plan for the next refresh.",
        }
    if pred <= 100:
        return {
            "outcome": "Borderline",
            "class": "decision-orange",
            "band": "85.6–100 g/km",
            "horizon": "Non-compliant now; marginal",
            "decision": "Conditional approval",
            "action": "Do not approve outright. Request independent emissions test from manufacturer. If test confirms prediction, reject or require a PHEV/BEV variant submission. Set 90-day deadline for manufacturer response.",
        }
    return {
        "outcome": "Non-compliant",
        "class": "decision-red",
        "band": "> 100 g/km",
        "horizon": "Non-compliant across all horizons",
        "decision": "Reject",
        "action": "Reject submission. Provide manufacturer with predicted CO₂, applicable threshold, and compliance gap in writing. Offer a structured re-submission pathway if manufacturer commits to a PHEV or BEV variant within 12 months.",
    }

def render_cluster_card(row: pd.Series, selected_cluster: Optional[str], button_key: str) -> None:
    color = CLUSTER_COLORS.get(row["cluster"], "#59a5ff")
    risk_label, risk_color = get_cluster_risk(row["cluster"])
    is_selected = row["cluster"] == selected_cluster
    selected_cls = " selected" if is_selected else ""

    html = f"""
    <div class='cluster-card-wrap' style='--cluster-color:{color};'>
      <div class='cluster-card{selected_cls}'>
        <div>
          <div class='cluster-topline'>
            <div class='cluster-type'>{row['dominant_powertrain']}</div>
            <div class='cluster-badge'></div>
          </div>
          <div class='cluster-name'>{row['cluster']}</div>
          <div class='cluster-risk-tag' style='background:{risk_color}22; border-color:{risk_color}44; color:{risk_color};'>{risk_label}</div>
        </div>
        <div class='cluster-metrics'>
          <div class='cluster-share'>{row['share']:.2f}%</div>
          <div class='cluster-count'>{int(row['count']):,} vehicles</div>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("<div class='cluster-action-wrap'>", unsafe_allow_html=True)
    if st.button("See details", key=button_key, use_container_width=True):
        st.session_state["selected_cluster"] = row["cluster"]
    st.markdown("</div>", unsafe_allow_html=True)

def get_safe_options(df: pd.DataFrame, col: str, fallback: List[object]) -> List[object]:
    if col in df.columns and df[col].dropna().shape[0] > 0:
        vals = sorted(df[col].dropna().astype(str).unique().tolist()) if df[col].dtype == object else sorted(df[col].dropna().unique().tolist())
        return vals
    return fallback


def _split_items(value: object) -> list[str]:
    if value is None:
        return []
    s = str(value).replace("<br>", ";").replace("\n", ";").strip()
    if not s or s.lower() == "nan":
        return []
    return [part.strip() for part in s.split(";") if part.strip()]

def _to_numbered_html(value: object) -> str:
    items = _split_items(value)
    if not items:
        return "-"
    return "<br>".join([f"{i+1}. {item}" for i, item in enumerate(items)])

def _parse_ranked_item(item: str) -> tuple[str, str]:
    s = str(item).strip()
    m = re.match(r"^(.*?)(\((.*?)\))?$", s)
    if not m:
        return s, ""
    name = (m.group(1) or "").strip().rstrip(";")
    meta = (m.group(3) or "").strip()
    return name, meta

def _to_profile_list_html(value: object, show_meta_inline: bool = True) -> str:
    items = _split_items(value)
    if not items:
        return "<div class='detail-item-name'>-</div>"
    rows = []
    for i, item in enumerate(items, start=1):
        name, meta = _parse_ranked_item(item)
        line = f"<span class='detail-rank'>{i}.</span><span class='detail-item-name'>{name}</span>"
        if meta and show_meta_inline:
            line += f" <span class='detail-item-meta'>({meta})</span>"
        rows.append(f"<div class='detail-list-item'>{line}</div>")
    return "<div class='detail-list-grid'>" + "".join(rows) + "</div>"

def _to_vehicle_class_profile_html(value: object) -> str:
    items = _split_items(value)
    if not items:
        return "<div class='detail-item-name'>-</div>"
    rows = []
    for i, item in enumerate(items, start=1):
        name, meta = _parse_ranked_item(item)
        meta_html = f"<span class='detail-item-inline-meta'>{meta}</span>" if meta else ""
        rows.append(
            "<div class='detail-list-item'>"
            f"<div class='detail-item-line'><div><span class='detail-rank'>{i}.</span><span class='detail-item-name'>{name}</span></div>{meta_html}</div>"
            "</div>"
        )
    return "<div class='detail-list-grid'>" + "".join(rows) + "</div>"

def _to_multiline_html(value: object) -> str:
    items = _split_items(value)
    if not items:
        return "-"
    return "<br>".join(items)




def _brand_portfolio_mix(sub: pd.DataFrame, top_n: int = 3) -> list[tuple[str, float]]:
    if "cluster" not in sub.columns or sub.empty:
        return []
    dist = (sub["cluster"].astype(str).value_counts(normalize=True).head(top_n) * 100).round(1)
    return [(idx, float(val)) for idx, val in dist.items()]

def _brand_core_issue(primary_cluster: str, avg_co2: float, bev_share: float, phev_share: float, target_2026: float) -> str:
    cluster = str(primary_cluster)
    if cluster == "Work Trucks":
        return f"Portfolio remains concentrated in work-truck formats, with average CO₂ around {avg_co2:.0f} g/km versus the 2026 threshold of {target_2026:.0f} g/km."
    if cluster == "Large SUVs":
        return f"High-emission SUV exposure dominates the brand portfolio, keeping average CO₂ around {avg_co2:.0f} g/km."
    if cluster == "Performance Cars":
        return f"Performance-oriented trims keep portfolio emissions elevated, with limited low-emission offset in the current mix."
    if bev_share < 0.02 and avg_co2 > target_2026 + 70:
        return f"The portfolio has very limited zero-emission hedge, while average CO₂ remains high at roughly {avg_co2:.0f} g/km."
    if phev_share >= 0.08:
        return f"The brand has started a transition through plug-in hybrid variants, but the ICE-heavy mix still leaves a substantial 2026 gap."
    return f"Current portfolio emissions are above the 2026 benchmark, especially in the brand’s core {cluster.lower()} offerings."

def _brand_recommended_action(primary_cluster: str, bev_share: float, phev_share: float) -> str:
    cluster = str(primary_cluster)
    if cluster == "Work Trucks":
        return "Prioritise lower-CO₂ truck trims and publish an electrified truck roadmap before the 2026 compliance milestone."
    if cluster == "Large SUVs":
        return "Shift flagship SUV lines toward HEV, PHEV, or BEV variants and reduce reliance on the highest-emission configurations."
    if cluster == "Performance Cars":
        return "Use premium positioning to accelerate electrified halo models and cap the highest-emission trims in new submissions."
    if bev_share < 0.02:
        return "Require at least one zero-emission or strong hybrid hedge within major nameplates before 2026 review."
    if phev_share >= 0.08:
        return "Convert transition variants into lower-CO₂ mainstream offers and tighten ICE baselines across the portfolio."
    return "Target the brand’s dominant cluster first and require a clearer compliance pathway across its highest-volume models."


def load_portfolio_risk_table() -> pd.DataFrame:
    rows = [
        {
            "brand": "Ford",
            "primary_cluster": "Work-Trucks",
            "total_models": 577,
            "non_compliant_2026": 577,
            "rate_2026": 100.0,
            "rate_2031": 100.0,
            "core_issue": "100% ICE; heaviest truck portfolio. Avg 261 g/km.",
            "recommended_action": "Mandate electrification roadmap for F-150 line within 2 years. Ford has the EV capacity—apply it here.",
        },
        {
            "brand": "Chevrolet",
            "primary_cluster": "Urban + Trucks",
            "total_models": 515,
            "non_compliant_2026": 515,
            "rate_2026": 100.0,
            "rate_2031": 100.0,
            "core_issue": "Spread across 4 clusters, all ICE. No BEV hedge.",
            "recommended_action": "Require cluster-specific compliance plans. Prioritise truck and large SUV segments where gap is widest.",
        },
        {
            "brand": "BMW",
            "primary_cluster": "Family Cars",
            "total_models": 690,
            "non_compliant_2026": 556,
            "rate_2026": 81.0,
            "rate_2031": 82.0,
            "core_issue": "Split portfolio: 115 BEVs help, but 501 ICE still non-compliant.",
            "recommended_action": "Leverage existing BEV infrastructure to accelerate ICE replacement in Family Cars cluster.",
        },
        {
            "brand": "Mercedes-Benz",
            "primary_cluster": "Family Cars",
            "total_models": 365,
            "non_compliant_2026": 365,
            "rate_2026": 100.0,
            "rate_2031": 100.0,
            "core_issue": "Premium brand, 3.8L avg engine. Zero-compliant models.",
            "recommended_action": "Engage on PHEV transition first—technical capacity exists. Set 2028 compliance milestone.",
        },
        {
            "brand": "GMC",
            "primary_cluster": "Work-Trucks",
            "total_models": 309,
            "non_compliant_2026": 289,
            "rate_2026": 94.0,
            "rate_2031": 94.0,
            "core_issue": "Large SUV and truck specialist, avg 281 g/km.",
            "recommended_action": "Co-engage with Ford/Chevrolet in sector-wide truck compliance programme. 3 brands = 72% of Work-Trucks.",
        },
        {
            "brand": "Porsche",
            "primary_cluster": "Family Cars",
            "total_models": 296,
            "non_compliant_2026": 296,
            "rate_2026": 100.0,
            "rate_2031": 100.0,
            "core_issue": "High avg CO₂ (259 g/km). Taycan EV line not in ICE catalogue.",
            "recommended_action": "Apply per-model CO₂ caps. Premium positioning means higher consumer tolerance for electrified versions.",
        },
        {
            "brand": "Toyota",
            "primary_cluster": "Urban Commuters",
            "total_models": 276,
            "non_compliant_2026": 276,
            "rate_2026": 100.0,
            "rate_2031": 100.0,
            "core_issue": "All ICE, avg 224 g/km despite hybrid reputation.",
            "recommended_action": "Hybrid technology exists—require active deployment across ICE model lines. Set 2027 HEV-minimum standard.",
        },
        {
            "brand": "Audi",
            "primary_cluster": "Urban Commuters",
            "total_models": 263,
            "non_compliant_2026": 263,
            "rate_2026": 100.0,
            "rate_2031": 100.0,
            "core_issue": "Avg 253 g/km; e-tron line not reflected in ICE catalogue.",
            "recommended_action": "Require PHEV or BEV variant for each new ICE model submission from 2027 onwards.",
        },
        {
            "brand": "Honda",
            "primary_cluster": "Family Cars",
            "total_models": 244,
            "non_compliant_2026": 221,
            "rate_2026": 91.0,
            "rate_2031": 91.0,
            "core_issue": "Portfolio remains dominated by ICE family and commuter models with limited zero-emission cover.",
            "recommended_action": "Require accelerated hybrid and BEV deployment across core family-car lines before the next product cycle.",
        },
        {
            "brand": "Nissan",
            "primary_cluster": "Urban Commuters",
            "total_models": 231,
            "non_compliant_2026": 210,
            "rate_2026": 91.0,
            "rate_2031": 91.0,
            "core_issue": "Strong EV signal exists, but too much of the portfolio remains above the 2026 threshold.",
            "recommended_action": "Prioritise electrified replacement of higher-emitting commuter and crossover variants from the next submission round.",
        },
    ]
    out = pd.DataFrame(rows)
    out["non_compliant_2031"] = out["non_compliant_2026"]
    out["avg_co2"] = out["core_issue"].str.extract(r"Avg\s*(\d+)").astype(float)
    out["avg_co2"] = out["avg_co2"].fillna(out["core_issue"].str.extract(r"avg\s*(\d+)").astype(float)[0])
    out["avg_co2"] = pd.to_numeric(out["avg_co2"], errors="coerce")
    out["portfolio_mix"] = "Illustrative policy summary table"
    out["bev_share_pct"] = np.nan
    out["phev_share_pct"] = np.nan
    return out.sort_values(["rate_2026", "non_compliant_2026", "total_models"], ascending=[False, False, False]).reset_index(drop=True)

def _brand_display_name(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        return s
    upper = s.upper()
    special = {
        "BMW": "BMW", "GMC": "GMC", "MINI": "MINI", "FIAT": "FIAT",
        "ALFA ROMEO": "Alfa Romeo", "ASTON MARTIN": "Aston Martin",
        "LAND ROVER": "Land Rover", "MERCEDES-BENZ": "Mercedes-Benz",
        "ROLLS-ROYCE": "Rolls-Royce", "LAMBORGHINI": "Lamborghini",
        "MASERATI": "Maserati", "PORSCHE": "Porsche", "AUDI": "Audi",
        "NISSAN": "Nissan", "TOYOTA": "Toyota", "CHEVROLET": "Chevrolet",
        "FORD": "Ford", "HONDA": "Honda", "ACURA": "Acura", "BUICK": "Buick",
        "CADILLAC": "Cadillac", "CHRYSLER": "Chrysler", "DODGE": "Dodge",
        "HYUNDAI": "Hyundai", "INFINITI": "Infiniti", "JAGUAR": "Jaguar",
        "JEEP": "Jeep", "KIA": "Kia", "LEXUS": "Lexus", "LINCOLN": "Lincoln",
        "MAZDA": "Mazda", "MITSUBISHI": "Mitsubishi", "RAM": "Ram",
        "RIVIAN": "Rivian", "SUBARU": "Subaru", "TESLA": "Tesla", "VOLKSWAGEN": "Volkswagen",
        "VOLVO": "Volvo", "GENESIS": "Genesis", "POLESTAR": "Polestar"
    }
    if upper in special:
        return special[upper]
    return upper.title()


def load_cluster_labeled_vehicle_table() -> pd.DataFrame:
    candidates = [
        Path('/mnt/data/vehicles_with_cluster_and_labels_v1 1.csv'),
        Path('/mnt/data/vehicles_with_cluster_and_labels_v1_1.csv'),
        Path(__file__).resolve().parent / 'vehicles_with_cluster_and_labels_v1 1.csv',
        Path(__file__).resolve().parent / 'vehicles_with_cluster_and_labels_v1_1.csv',
        Path(__file__).resolve().parent / 'data' / 'dashboard' / 'vehicles_with_cluster_and_labels_v1 1.csv',
        Path(__file__).resolve().parent / 'data' / 'dashboard' / 'vehicles_with_cluster_and_labels_v1_1.csv',
        Path(__file__).resolve().parent.parent / 'data' / 'dashboard' / 'vehicles_with_cluster_and_labels_v1 1.csv',
        Path(__file__).resolve().parent.parent / 'data' / 'dashboard' / 'vehicles_with_cluster_and_labels_v1_1.csv',
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return pd.DataFrame()



def load_brand_core_issue_table() -> pd.DataFrame:
    candidates = [
        Path('/mnt/data/market_overview_data_with_brand_core_issues.xlsx'),
        Path('/mnt/data/market_overview_data.xlsx'),
        Path(__file__).resolve().parent / 'market_overview_data_with_brand_core_issues.xlsx',
        Path(__file__).resolve().parent / 'market_overview_data.xlsx',
        Path(__file__).resolve().parent / 'data' / 'dashboard' / 'market_overview_data_with_brand_core_issues.xlsx',
        Path(__file__).resolve().parent / 'data' / 'dashboard' / 'market_overview_data.xlsx',
        Path(__file__).resolve().parent.parent / 'data' / 'dashboard' / 'market_overview_data_with_brand_core_issues.xlsx',
        Path(__file__).resolve().parent.parent / 'data' / 'dashboard' / 'market_overview_data.xlsx',
    ]
    for p in candidates:
        if p.exists():
            try:
                sheet_names = pd.ExcelFile(p).sheet_names
                if "brand_core_issues" not in sheet_names:
                    continue
                core = pd.read_excel(p, sheet_name="brand_core_issues")
                if {"Brand", "Core issue"}.issubset(set(core.columns)):
                    core = core.rename(columns={"Brand": "brand", "Core issue": "core_issue"})
                    core["brand"] = core["brand"].astype(str).str.strip()
                    core = core.loc[core["brand"].notna() & (core["brand"] != "")].copy()
                    return core[["brand", "core_issue"]]
            except Exception:
                continue
    return pd.DataFrame(columns=["brand", "core_issue"])


def build_brand_compliance_summary_from_cluster_file(cluster_df: pd.DataFrame, target_schedule: Dict[int, float]) -> pd.DataFrame:
    expected_cols = [
        "brand", "primary_cluster", "total_models", "non_compliant_2026", "non_compliant_2031",
        "rate_2026", "rate_2031", "avg_co2", "core_issue", "recommended_action"
    ]
    if cluster_df.empty:
        return pd.DataFrame(columns=expected_cols)

    required = {"Make", "vehicle_id", "CO2_Emissions_g_km", "Commercial_Label"}
    if not required.issubset(set(cluster_df.columns)):
        return pd.DataFrame(columns=expected_cols)

    work = cluster_df.copy()
    work = work.loc[work["Make"].notna() & work["vehicle_id"].notna()].copy()
    work["brand_key"] = work["Make"].astype(str).str.strip().str.upper()
    work = work.loc[work["brand_key"] != ""].copy()
    work["brand"] = work["brand_key"].map(_brand_display_name)
    work["co2"] = pd.to_numeric(work["CO2_Emissions_g_km"], errors="coerce")
    work["cluster_label"] = work["Commercial_Label"].astype(str).str.strip()

    target_2026 = float(target_schedule.get(2026, 86))
    target_2031 = float(target_schedule.get(2031, target_2026))

    # approximate electrification shares for recommendation text
    vehicle_type_col = None
    for c in ["Vehicle_Type", "Fuel_Type_Primary", "Transmission_Type"]:
        if c in work.columns:
            vehicle_type_col = c
            break

    rows = []
    for brand_key, sub in work.groupby("brand_key", dropna=False):
        total_models = int(len(sub))
        if total_models == 0:
            continue
        co2 = sub["co2"].dropna()
        non_2026 = int((sub["co2"] > target_2026).fillna(False).sum())
        non_2031 = int((sub["co2"] > target_2031).fillna(False).sum())
        rate_2026 = non_2026 / total_models * 100
        rate_2031 = non_2031 / total_models * 100
        avg_co2 = float(co2.mean()) if not co2.empty else float('nan')
        primary_cluster = str(sub["cluster_label"].mode().iloc[0]) if sub["cluster_label"].notna().any() else "Unknown"

        bev_share = float((sub["co2"].fillna(np.inf) == 0).mean())
        phev_share = 0.0
        if vehicle_type_col is not None:
            phev_share = float(sub[vehicle_type_col].astype(str).str.upper().str.contains('PHEV').mean())

        rows.append({
            "brand": _brand_display_name(brand_key),
            "primary_cluster": primary_cluster,
            "total_models": total_models,
            "non_compliant_2026": non_2026,
            "non_compliant_2031": non_2031,
            "rate_2026": rate_2026,
            "rate_2031": rate_2031,
            "avg_co2": avg_co2,
            "core_issue": _brand_core_issue(primary_cluster, avg_co2 if pd.notna(avg_co2) else target_2026, bev_share, phev_share, target_2026),
            "recommended_action": _brand_recommended_action(primary_cluster, bev_share, phev_share),
        })

    if not rows:
        return pd.DataFrame(columns=expected_cols)

    result = pd.DataFrame(rows).sort_values(["rate_2026", "non_compliant_2026", "total_models", "brand"], ascending=[False, False, False, True]).reset_index(drop=True)
    return result


def build_brand_compliance_summary(df: pd.DataFrame, target_schedule: Dict[int, float]) -> pd.DataFrame:
    expected_cols = [
        "brand", "primary_cluster", "total_models", "non_compliant_2026", "non_compliant_2031",
        "rate_2026", "rate_2031", "avg_co2", "bev_share_pct", "phev_share_pct",
        "portfolio_mix", "core_issue", "recommended_action"
    ]
    if df.empty or "make" not in df.columns or "co2" not in df.columns:
        return pd.DataFrame(columns=expected_cols)
    target_2026 = float(target_schedule.get(2026, 86))
    target_2031 = float(target_schedule.get(2031, target_2026))
    work = df.copy()
    work = work.loc[work["make"].notna() & work["co2"].notna()].copy()
    work["make"] = work["make"].astype(str).str.strip()
    work = work.loc[work["make"] != ""].copy()
    rows = []
    for brand, sub in work.groupby("make"):
        total_models = int(len(sub))
        if total_models == 0:
            continue
        non_2026 = int((sub["co2"].astype(float) > target_2026).sum())
        non_2031 = int((sub["co2"].astype(float) > target_2031).sum())
        rate_2026 = non_2026 / total_models * 100
        rate_2031 = non_2031 / total_models * 100
        primary_cluster = str(sub["cluster"].astype(str).value_counts().index[0]) if "cluster" in sub.columns and not sub["cluster"].dropna().empty else "Unknown"
        avg_co2 = float(sub["co2"].mean())
        bev_share = float((sub["powertrain"].astype(str) == "BEV").mean()) if "powertrain" in sub.columns else 0.0
        phev_share = float((pd.to_numeric(sub["phev_flag"], errors="coerce").fillna(0) == 1).mean()) if "phev_flag" in sub.columns else 0.0
        mix = _brand_portfolio_mix(sub)
        mix_text = " · ".join([f"{name} {share:.0f}%" for name, share in mix]) if mix else "Not available"
        rows.append({
            "brand": brand,
            "primary_cluster": primary_cluster,
            "total_models": total_models,
            "non_compliant_2026": non_2026,
            "non_compliant_2031": non_2031,
            "rate_2026": rate_2026,
            "rate_2031": rate_2031,
            "avg_co2": avg_co2,
            "bev_share_pct": bev_share * 100,
            "phev_share_pct": phev_share * 100,
            "portfolio_mix": mix_text,
            "core_issue": _brand_core_issue(primary_cluster, avg_co2, bev_share, phev_share, target_2026),
            "recommended_action": _brand_recommended_action(primary_cluster, bev_share, phev_share),
        })
    if not rows:
        return pd.DataFrame(columns=expected_cols)
    result = pd.DataFrame(rows).sort_values(["rate_2026", "non_compliant_2026", "total_models"], ascending=[False, False, False])
    return result.reset_index(drop=True)

def make_brand_exposure_chart(brands: pd.DataFrame, selected_brand: Optional[str] = None) -> go.Figure:
    work = brands.copy().sort_values("rate_2026", ascending=True)
    colors = []
    for _, row in work.iterrows():
        if row["brand"] == selected_brand:
            colors.append("#ff7a84")
        elif row["rate_2026"] >= 100:
            colors.append("#ff8a94")
        elif row["rate_2026"] >= 80:
            colors.append("#ffb020")
        else:
            colors.append("#7cb7ff")
    fig = go.Figure(go.Bar(
        x=work["rate_2026"],
        y=work["brand"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#f6f8ff", width=[2.2 if b == selected_brand else 0 for b in work["brand"]])),
        text=[f"{v:.0f}%" for v in work["rate_2026"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Non-compliant share: %{x:.1f}%<br><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top brands by 2026 portfolio risk", font=dict(size=26, color="#f4f7fb")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        font=dict(color="#f4f7fb", size=14),
        margin=dict(l=20, r=30, t=78, b=35),
        height=760,
        showlegend=False,
        xaxis=dict(title="% of total portfolio non-compliant in 2026", range=[0, min(110, max(100, float(work["rate_2026"].max())*1.12 if not work.empty else 100))], gridcolor="rgba(255,255,255,0.08)", ticksuffix="%"),
        yaxis=dict(title="", automargin=True),
    )
    return fig
# ============================
# Sidebar / state
# ============================
if "selected_cluster" not in st.session_state:
    st.session_state["selected_cluster"] = CLUSTER_ORDER[0]
if "selected_brand" not in st.session_state:
    st.session_state["selected_brand"] = None

st.sidebar.markdown("## Dashboard controls")
page = st.sidebar.radio("Navigate", ["Home", "Market", "Portfolio Risk", "Calculator"], label_visibility="collapsed")
custom_path = None

try:
    df, data_path, mapping = load_data(custom_path)
    model = load_model()
    market_pkg, market_pkg_path = load_market_package()
    target_schedule = load_target_schedule_from_workbook()
    if market_pkg is not None and not market_pkg["summary"].empty:
        summary = market_pkg["summary"].copy()
    else:
        summary = cluster_summary(df)
    if st.session_state["selected_cluster"] not in summary["cluster"].tolist() and not summary.empty:
        st.session_state["selected_cluster"] = summary.iloc[0]["cluster"]
    cluster_brand_source = load_cluster_labeled_vehicle_table()
    if not cluster_brand_source.empty:
        brand_summary = build_brand_compliance_summary_from_cluster_file(cluster_brand_source, target_schedule)
    else:
        brand_summary = build_brand_compliance_summary(df, target_schedule)
        if brand_summary.empty:
            brand_summary = load_portfolio_risk_table()
    brand_core_issue_table = load_brand_core_issue_table()
    if not brand_summary.empty and not brand_core_issue_table.empty:
        brand_summary = brand_summary.merge(
            brand_core_issue_table.drop_duplicates("brand"),
            on="brand",
            how="left",
            suffixes=("", "_from_sheet")
        )
        if "core_issue_from_sheet" in brand_summary.columns:
            brand_summary["core_issue"] = brand_summary["core_issue_from_sheet"].where(
                brand_summary["core_issue_from_sheet"].notna() & (brand_summary["core_issue_from_sheet"].astype(str).str.strip() != ""),
                brand_summary.get("core_issue")
            )
            brand_summary = brand_summary.drop(columns=["core_issue_from_sheet"])
    if not brand_summary.empty and "brand" in brand_summary.columns:
        if st.session_state["selected_brand"] not in brand_summary["brand"].tolist():
            st.session_state["selected_brand"] = sorted(brand_summary["brand"].tolist())[0]
    else:
        st.session_state["selected_brand"] = None
    data_ok = True
except Exception as e:
    data_ok = False
    df = pd.DataFrame()
    summary = pd.DataFrame()
    data_path = None
    market_pkg = None
    market_pkg_path = None
    model = None
    mapping = {}
    target_schedule = DEFAULT_TARGET_SCHEDULE.copy()
    brand_summary = pd.DataFrame()
    cluster_brand_source = pd.DataFrame()
    brand_core_issue_table = pd.DataFrame()
    st.error(str(e))


selected_cluster = st.session_state.get("selected_cluster", CLUSTER_ORDER[0])

# ============================
# Pages
# ============================
if not data_ok:
    st.stop()

if page == "Home":
    year_min = int(df["model_year"].min()) if "model_year" in df.columns and df["model_year"].notna().any() else 2012
    year_max = int(df["model_year"].max()) if "model_year" in df.columns and df["model_year"].notna().any() else 2026
    non_empty = summary.shape[0]

    st.markdown(
        """
        <div class="home-hero-shell">
            <div class="home-hero-grid">
                <div>
                    <div class="home-hero-title">ComplianceIQ</div>
                    <div class="home-hero-subtitle">Market intelligence for vehicle emissions compliance</div>
                    <div class="home-hero-supporting">Evaluate proposed vehicle models against Canadian market patterns and model-year CO₂ targets.</div>
                </div>
                <div class="home-badges-panel">
                    <div class="home-badge home-badge-a">Evidence-based</div>
                    <div class="home-badge home-badge-b">Market-benchmarked</div>
                    <div class="home-badge home-badge-c">Yearly compliance</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="home-section-card">
            <div class="home-two-col">
                <div>
                    <div class="home-col-title">What does it do?</div>
                    <div class="home-col-accent home-col-accent-a"></div>
                    <div class="home-bullet-list">
                        <div class="home-bullet-item"><span class="home-bullet-dot home-bullet-dot-a"></span><span><span class="bullet-key-blue">Segments</span> the Canadian vehicle market</span></div>
                        <div class="home-bullet-item"><span class="home-bullet-dot home-bullet-dot-a"></span><span><span class="bullet-key-teal">Estimates CO₂</span> for proposed models</span></div>
                        <div class="home-bullet-item"><span class="home-bullet-dot home-bullet-dot-a"></span><span><span class="bullet-key-amber">Checks compliance</span> by model year</span></div>
                    </div>
                </div>
                <div>
                    <div class="home-col-title">Why is it important?</div>
                    <div class="home-col-accent home-col-accent-b"></div>
                    <div class="home-bullet-list">
                        <div class="home-bullet-item"><span class="home-bullet-dot home-bullet-dot-b"></span><span><span class="bullet-key-blue">Supports</span> more consistent screening</span></div>
                        <div class="home-bullet-item"><span class="home-bullet-dot home-bullet-dot-b"></span><span><span class="bullet-key-teal">Benchmarks</span> against real market patterns</span></div>
                        <div class="home-bullet-item"><span class="home-bullet-dot home-bullet-dot-b"></span><span><span class="bullet-key-amber">Provides</span> a clear yearly basis for assessment</span></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow-strip">
            <div class="workflow-text">Map the market <span class="workflow-arrow">→</span> Screen a submission <span class="workflow-arrow">→</span> Support approval review</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="summary-shell">
            <div class="summary-metrics">
                <div class="summary-metric">
                    <div class="summary-metric-label">Records loaded</div>
                    <div class="summary-metric-value metric-blue">{len(df):,}</div>
                    <div class="summary-metric-desc">Vehicle entries available for market analysis.</div>
                </div>
                <div class="summary-metric">
                    <div class="summary-metric-label">Year coverage</div>
                    <div class="summary-metric-value metric-teal">{year_min}–{year_max}</div>
                    <div class="summary-metric-desc">Observed model-year range in the current dataset.</div>
                </div>
                <div class="summary-metric">
                    <div class="summary-metric-label">Market coverage</div>
                    <div class="summary-metric-value metric-amber">{non_empty} clusters</div>
                    <div class="summary-metric-desc">Clustered market view used for benchmarking and screening.</div>
                </div>
            </div>
            <div class="policy-note-inline">
                <span class="policy-note-icon">ⓘ</span>
                <span>Advisory tool for screening and review; not a replacement for formal certification.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Market":
    st.markdown("<div class='page-title'>Market Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Processed Canadian vehicle data are grouped into 8 updated clusters representing different powertrain, size, and emissions profiles.</div>",
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([4.8, 2.0], gap="large")
    with top_left:
        st.markdown("<div class='subsection-title'>Cluster overview</div>", unsafe_allow_html=True)
        card_rows = [CLUSTER_ORDER[0:2], CLUSTER_ORDER[2:4], CLUSTER_ORDER[4:6], CLUSTER_ORDER[6:8]]
        for row_idx, row_clusters in enumerate(card_rows):
            cols = st.columns(2, gap="large")
            for i, cluster_name in enumerate(row_clusters):
                cluster_row = summary[summary["cluster"] == cluster_name]
                with cols[i]:
                    if cluster_row.empty:
                        st.markdown("<div class='cluster-card'><div class='cluster-name'>No data</div></div>", unsafe_allow_html=True)
                    else:
                        render_cluster_card(cluster_row.iloc[0], selected_cluster, f"cluster_btn_{row_idx}_{i}")
    with top_right:
        st.markdown("<div class='subsection-title'>Market share</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title' style='margin-bottom: 0.85rem;'>Share of Full Market</div>", unsafe_allow_html=True)
        donut_source = market_pkg["market_share"].copy() if market_pkg is not None else summary[["cluster", "count", "share"]].copy()
        donut = make_donut(donut_source, selected_cluster, title=None)
        donut.update_layout(height=500, margin=dict(l=5, r=5, t=0, b=10))
        st.plotly_chart(donut, use_container_width=True, config={"displaylogo": False})

    st.markdown("<div class='subsection-title'>Selected cluster detail</div>", unsafe_allow_html=True)
    srow = summary[summary["cluster"] == selected_cluster].iloc[0]
    top_makes_html = _to_profile_list_html(srow["top_makes"])
    top_models_html = _to_profile_list_html(srow["top_models"])
    top_classes_html = _to_vehicle_class_profile_html(srow["top_classes"])
    cluster_desc = str(srow.get("description", "")).strip() if pd.notna(srow.get("description", "")) else ""
    if not cluster_desc:
        cluster_desc = "No summary description available for this cluster."

    risk_label, risk_color = get_cluster_risk(srow['cluster'])
    left_col, right_col = st.columns([1.0, 1.45], gap="large")
    with left_col:
        st.markdown(f'''
        <div class="detail-card primary-profile">
            <div class="detail-section-label">Cluster profile</div>
            <div class="detail-primary-title">{srow['cluster']}</div>
            <div class="detail-risk-pill" style="background:{risk_color}22; border-color:{risk_color}44; color:{risk_color};">{risk_label}</div>
            <div class="detail-summary">{cluster_desc}</div>
            <div class="detail-pill-row">
                <div class="detail-mini-pill">
                    <div class="detail-pill-label">Dominant power</div>
                    <div class="detail-pill-value">{srow['dominant_powertrain']}</div>
                </div>
                <div class="detail-mini-pill">
                    <div class="detail-pill-label">Most common fuel</div>
                    <div class="detail-pill-value">{srow['common_fuel']}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="detail-card secondary-profile">
            <div class="detail-section-label">Top vehicle classes</div>
            <div class="detail-value detail-list-compact">{top_classes_html}</div>
        </div>
        ''', unsafe_allow_html=True)
    with right_col:
        st.markdown(f'''
        <div class="detail-card market-leaders">
            <div class="detail-section-label">Market leaders</div>
            <div class="detail-two-col">
                <div class="detail-col-block">
                    <div class="detail-subgroup-title">Top 5 makes by count</div>
                    <div class="detail-value detail-list-compact">{top_makes_html}</div>
                </div>
                <div class="detail-col-block">
                    <div class="detail-subgroup-title">Top 5 models by count</div>
                    <div class="detail-value detail-list-compact">{top_models_html}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<div class='subsection-title'>Cluster comparison across the full market</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="large")
    with g1:
        if market_pkg is not None:
            st.plotly_chart(
                make_boxplot_from_precomputed(
                    market_pkg["boxstats"],
                    "co2",
                    "CO₂ emissions distribution across clusters",
                    "CO₂ emissions (g/km)",
                    selected_cluster,
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.plotly_chart(
                make_boxplot(df, "co2", "CO₂ emissions distribution across clusters", "CO₂ emissions (g/km)", selected_cluster),
                use_container_width=True,
                config={"displaylogo": False},
            )
    with g2:
        if market_pkg is not None:
            st.plotly_chart(
                make_boxplot_from_precomputed(
                    market_pkg["boxstats"],
                    "engine_size",
                    "Engine size distribution across clusters",
                    "Engine size (L)",
                    selected_cluster,
                    note=None,
                ),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.plotly_chart(
                make_boxplot(df, "engine", "Engine size distribution across clusters", "Engine size (L)", selected_cluster),
                use_container_width=True,
                config={"displaylogo": False},
            )

    bottom_cols = st.columns(2, gap="large")
    with bottom_cols[0]:
        st.plotly_chart(
            make_bar(summary, "avg_co2", "Average CO₂ by cluster", "Average CO₂ emissions (g/km)", selected_cluster),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with bottom_cols[1]:
        if summary["avg_energy"].notna().any():
            st.plotly_chart(
                make_bar(summary, "avg_energy", "Average energy consumption by cluster", "Average energy consumption (MJ/100 km)", selected_cluster),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info("Energy consumption data are not available in the current dataset.")



elif page == "Portfolio Risk":
    st.markdown("<div class='page-title'>Portfolio Risk</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Brand-level portfolio exposure recalculated from the cluster-labelled vehicle table for 2026 and 2031 compliance thresholds.</div>",
        unsafe_allow_html=True,
    )

    target_2026 = float(target_schedule.get(2026, 86))
    target_2031 = float(target_schedule.get(2031, target_2026))
    brand_all = brand_summary.copy() if not brand_summary.empty else pd.DataFrame()
    brand_top10 = brand_all.head(10).copy() if not brand_all.empty else pd.DataFrame()

    if brand_top10.empty:
        st.info("Portfolio risk view could not be populated from the uploaded cluster-labelled vehicle table.")
    else:
        highest = brand_top10.iloc[0]
        fully_exposed = int((brand_all["rate_2026"] >= 99.95).sum())

        k1, k2 = st.columns(2, gap="large")
        with k1:
            st.markdown(f"""
            <div class="brand-kpi-card">
                <div class="brand-kpi-label">Highest exposure</div>
                <div class="brand-kpi-value accent-red">{highest['brand']}</div>
                <div class="brand-kpi-desc">{int(highest['non_compliant_2026']):,} of {int(highest['total_models']):,} model entries breach the 2026 rule ({highest['rate_2026']:.0f}%).</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="brand-kpi-card">
                <div class="brand-kpi-label">Fully exposed brands</div>
                <div class="brand-kpi-value accent-blue">{fully_exposed}</div>
                <div class="brand-kpi-desc">Brands whose entire portfolio breaches the 2026 threshold.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.2, 1.0], gap="large")

        with left:
            chart_df = brand_top10.copy()
            st.plotly_chart(
                make_brand_exposure_chart(chart_df, st.session_state.get("selected_brand")),
                use_container_width=True,
                config={"displaylogo": False},
            )

        with right:
            brand_options = sorted(brand_all["brand"].tolist()) if not brand_all.empty else sorted(chart_df["brand"].tolist())
            chosen = st.selectbox(
                "Select a brand for interpretation",
                brand_options,
                index=max(0, brand_options.index(st.session_state["selected_brand"])) if st.session_state.get("selected_brand") in brand_options else 0,
            )
            st.session_state["selected_brand"] = chosen
            source_df = brand_all if not brand_all.empty else chart_df
            brow = source_df.loc[source_df["brand"] == st.session_state["selected_brand"]].iloc[0]
            avg_co2_value = pd.to_numeric(pd.Series([brow.get("avg_co2", np.nan)]), errors="coerce").iloc[0]
            avg_co2_text = f"{float(avg_co2_value):.0f} g/km" if pd.notna(avg_co2_value) else "Not stated"

            st.markdown(f"""
            <div class="brand-detail-shell profile-main">
                <div class="brand-section-label">Selected brand profile</div>
                <div class="brand-detail-title">{brow['brand']}</div>
                <div class="brand-detail-sub">{brow['brand']} has <b>{int(brow['non_compliant_2026'])}</b> non-compliant model entries out of <b>{int(brow['total_models'])}</b> in the current portfolio under the 2026 threshold of {target_2026:.0f} g/km.</div>
                <div class="brand-pill-row">
                    <div class="brand-pill"><span class="meta">2026 rate</span> {brow['rate_2026']:.0f}%</div>
                    <div class="brand-pill"><span class="meta">2031 rate</span> {brow['rate_2031']:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="brand-detail-shell profile-mini">
                <div class="brand-grid-two">
                    <div class="brand-mini-card">
                        <div class="brand-mini-label">Total models</div>
                        <div class="brand-mini-value">{int(brow['total_models']):,}</div>
                    </div>
                    <div class="brand-mini-card">
                        <div class="brand-mini-label">Non-compliant @2026</div>
                        <div class="brand-mini-value">{int(brow['non_compliant_2026']):,}</div>
                    </div>
                    <div class="brand-mini-card">
                        <div class="brand-mini-label">Average CO₂</div>
                        <div class="brand-mini-value">{avg_co2_text}</div>
                    </div>
                    <div class="brand-mini-card">
                        <div class="brand-mini-label">Primary cluster</div>
                        <div class="brand-mini-value">{brow['primary_cluster']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="brand-detail-shell profile-action">
                <div class="brand-section-label">Core issue</div>
                <div class="brand-issue-text">{brow['core_issue']}</div>
            </div>
            """, unsafe_allow_html=True)

elif page == "Calculator":
    st.markdown("<div class='page-title'>Approval Screening Console</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Estimate tailpipe CO₂ and check model-year compliance for a proposed vehicle submission.</div>",
        unsafe_allow_html=True,
    )

    model_features = get_model_features(model)
    year_options = sorted(y for y in target_schedule.keys() if y <= 2050)

    POWERTRAIN_OPTIONS = [
        "Battery electric vehicle (BEV)",
        "Plug-in hybrid electric vehicle (PHEV)",
        "Conventional / hybrid vehicle",
    ]
    VEHICLE_CLASS_OPTIONS = [
        "Small car",
        "Mid / full-size car",
        "SUV / pickup",
        "Van / utility",
        "Other",
    ]
    TRANSMISSION_OPTIONS = [
        "Automatic",
        "Automated manual",
        "Continuously variable transmission",
        "Manual",
        "Other / unknown",
    ]
    FUEL_GROUP_OPTIONS = [
        "Fossil fuel",
        "Alternative fuel",
        "Plug-in hybrid electric",
    ]

    vehicle_class_map = {
        "Small car": "small",
        "Mid / full-size car": "mid",
        "SUV / pickup": "suv_truck",
        "Van / utility": "van",
        "Other": "other",
    }
    transmission_map = {
        "Automatic": "automatic",
        "Automated manual": "automated_manual",
        "Continuously variable transmission": "cvt",
        "Manual": "manual",
        "Other / unknown": "other",
    }
    fuel_group_map = {
        "Fossil fuel": "fossil",
        "Alternative fuel": "alt_fuel",
        "Plug-in hybrid electric": "phev_elec",
    }

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        st.markdown("<div class='subsection-title' style='margin-top:0.2rem;'>Submission details</div>", unsafe_allow_html=True)

        st.markdown("#### Vehicle powertrain")
        powertrain_type = st.selectbox(
            "Powertrain type",
            POWERTRAIN_OPTIONS,
            help="Top-level powertrain category used to route the screening logic.",
        )

        st.markdown("#### Vehicle classification")
        vehicle_class_ui = st.selectbox(
            "Vehicle class",
            VEHICLE_CLASS_OPTIONS,
            help="Regulatory size grouping used by the model.",
        )
        transmission_ui = st.selectbox(
            "Transmission type",
            TRANSMISSION_OPTIONS,
            help="Consolidated transmission category used for screening.",
        )

        if powertrain_type == "Battery electric vehicle (BEV)":
            st.selectbox(
                "Primary fuel group",
                ["Battery electric drive"],
                index=0,
                disabled=True,
                help="BEV submissions are handled through rule-based zero tailpipe CO₂ treatment.",
            )
            selected_fuel_ui = None
        elif powertrain_type == "Plug-in hybrid electric vehicle (PHEV)":
            st.selectbox(
                "Primary fuel group",
                ["Plug-in hybrid electric"],
                index=0,
                disabled=True,
                help="Locked for plug-in hybrid submissions.",
            )
            selected_fuel_ui = "Plug-in hybrid electric"
        else:
            selected_fuel_ui = st.selectbox(
                "Primary fuel group",
                ["Fossil fuel", "Alternative fuel"],
                help="Broad fuel grouping used in emissions prediction.",
            )

        st.markdown("#### Powertrain and efficiency")
        engine_default = float(df["engine"].median()) if "engine" in df.columns and df["engine"].notna().any() else 2.0
        cyl_default = int(round(float(df["cylinders"].median()))) if "cylinders" in df.columns and df["cylinders"].notna().any() else 4
        fuel_cons_default = float(df["fuel_cons"].median()) if "fuel_cons" in df.columns and df["fuel_cons"].notna().any() else 7.5

        bev_mode = powertrain_type == "Battery electric vehicle (BEV)"
        phev_mode = powertrain_type == "Plug-in hybrid electric vehicle (PHEV)"

        engine_size = st.number_input(
            "Engine size (L)",
            min_value=0.0,
            max_value=8.0,
            value=0.0 if bev_mode else round(engine_default, 2),
            step=0.1,
            disabled=bev_mode,
        )
        cylinders = st.number_input(
            "Cylinders",
            min_value=0,
            max_value=12,
            value=0 if bev_mode else int(cyl_default),
            step=1,
            disabled=bev_mode,
        )
        fuel_cons = st.number_input(
            "Combined fuel consumption (L/100 km)",
            min_value=0.0,
            max_value=25.0,
            value=0.0 if bev_mode else round(fuel_cons_default, 2),
            step=0.1,
            disabled=bev_mode,
        )

        st.markdown("#### Regulatory context")
        model_year = st.selectbox("Model year", year_options, index=len(year_options)-1)

    vehicle_class_model = vehicle_class_map[vehicle_class_ui]
    transmission_model = transmission_map[transmission_ui]
    fuel_group_model = fuel_group_map.get(selected_fuel_ui or "", "fossil")
    is_phev = 1 if phev_mode else 0

    inputs = {
        "Model_Year": model_year,
        "Engine_Size_L": engine_size,
        "Cylinders": cylinders,
        "Fuel_Type_Primary": fuel_group_model,
        "Vehicle_Class": vehicle_class_model,
        "Transmission": transmission_model,
        "is_phev": is_phev,
        "Fuel_Cons_Comb_L100km": fuel_cons,
    }
    target = target_schedule[int(model_year)]
    if bev_mode:
        pred = 0.0
    else:
        pred = model_predict(model, inputs, model_features)
    margin = round(target - pred, 1)
    status_text, status_class = compliance_status(pred, target)

    with right:
        st.markdown("<div class='subsection-title' style='margin-top:0.2rem;'>Output</div>", unsafe_allow_html=True)
        r1, r2 = st.columns(2, gap="medium")
        with r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-kicker">Predicted CO₂ emissions</div>
                <div class="result-main">{pred:.1f} g/km</div>
                <div class="result-sub">{('Rule-based zero tailpipe treatment for BEV submissions.' if bev_mode else ('Estimated from loaded regression model.' if model is not None else 'Estimated from fallback transparent estimator.'))}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-kicker">Applicable CO₂ target</div>
                <div class="result-main">{target:.0f} g/km</div>
                <div class="result-sub">Year-specific target for model year {model_year}.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        r3, r4 = st.columns(2, gap="medium")
        with r3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-kicker">Selected model year</div>
                <div class="result-main">{model_year}</div>
                <div class="result-sub">Compliance is judged only for the selected year.</div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-kicker">Compliance result</div>
                <div class="status-pill {status_class}">{status_text}</div>
                <div class="result-sub">Margin to target: {margin:+.1f} g/km</div>
            </div>
            """, unsafe_allow_html=True)


        decision_info = classify_emissions_outcome(pred)
        st.markdown(f"""
        <div class="note-card" style="margin-top:1rem;">
            <div style="font-weight:800; font-size:1.08rem; margin-bottom:0.55rem;">Decision guidance</div>
            <div class="decision-label">Decision</div>
            <div class="decision-outcome {decision_info['class']}" style="margin-bottom:0.7rem;">{decision_info['decision']}</div>
            <div class="decision-label">What to do</div>
            <div class="decision-value">{decision_info['action']}</div>
        </div>
        """, unsafe_allow_html=True)


