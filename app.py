import warnings
warnings.filterwarnings("ignore")

# ── SSL 우회: 환경변수로 먼저 끄고 (Linux/Streamlit Cloud 대응)
#    curl_cffi가 임포트되기 전에 설정해야 함
import os
os.environ["CURL_CA_BUNDLE"]    = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"]     = ""

# ── SSL 우회: curl_cffi Session 레벨에서도 verify=False (Windows 대응)
try:
    import curl_cffi.requests as _curl_requests
    _orig_session_init = _curl_requests.Session.__init__
    def _patched_session_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_session_init(self, *args, **kwargs)
    _curl_requests.Session.__init__ = _patched_session_init
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import platform
import json
from pathlib import Path

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
import matplotlib.font_manager as _fm

def _set_korean_font():
    if platform.system() == "Windows":
        # Windows: 맑은 고딕
        rcParams["font.family"] = "Malgun Gothic"
        return

    # Linux(Streamlit Cloud): 나눔고딕 직접 등록
    # packages.txt 에 fonts-nanum 추가 시 아래 경로에 설치됨
    _nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if Path(_nanum_path).exists():
        _fm.fontManager.addfont(_nanum_path)
        _prop = _fm.FontProperties(fname=_nanum_path)
        rcParams["font.family"] = _prop.get_name()
        return

    # 시스템에 설치된 Nanum 계열 폰트 자동 탐색
    _candidates = [f.fname for f in _fm.fontManager.ttflist
                   if "nanum" in f.name.lower() or "gothic" in f.name.lower()]
    if _candidates:
        _fm.fontManager.addfont(_candidates[0])
        _prop = _fm.FontProperties(fname=_candidates[0])
        rcParams["font.family"] = _prop.get_name()
        return

    # 폴백: DejaVu Sans (한글 깨짐 방지용 경고)
    rcParams["font.family"] = "DejaVu Sans"

_set_korean_font()
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 120  # 모바일 레티나 대응

# ── 페이지 설정 (centered = 모바일에서 전체 너비 활용) ─────────────────────────
st.set_page_config(
    page_title="환율 예측",
    page_icon="💱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── 모바일 최적화 CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── 기본 여백 축소 ── */
.block-container {
    padding-top: 1rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    max-width: 100% !important;
}

/* ── 사이드바 숨김 ── */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── 앱 제목 ── */
.app-title {
    font-size: 22px;
    font-weight: 800;
    color: #1e3a5f;
    margin: 0 0 2px 0;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 12px;
}

/* ── 섹션 헤더 ── */
.section-header {
    font-size: 15px;
    font-weight: 700;
    color: #1e3a5f;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
    margin: 16px 0 8px;
}

/* ── 지표 카드 (2x2 그리드용) ── */
.card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 12px;
}
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 12px;
    padding: 12px 10px;
    color: white;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    opacity: 0.85;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
}
.metric-value {
    font-size: 18px;
    font-weight: 800;
    line-height: 1.2;
    word-break: break-all;
}
.metric-delta {
    font-size: 11px;
    margin-top: 3px;
    opacity: 0.9;
}

/* ── 입력 요소 크기 (손가락 터치 최적화) ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {
    min-height: 48px !important;
    font-size: 15px !important;
}
div[data-testid="stDateInput"] input {
    min-height: 48px !important;
    font-size: 15px !important;
}
div[data-testid="stRadio"] label {
    font-size: 15px !important;
    padding: 6px 4px !important;
}
div[data-testid="stCheckbox"] label {
    font-size: 15px !important;
}

/* ── 버튼 크게 ── */
.stButton > button {
    width: 100%;
    min-height: 52px;
    font-size: 16px !important;
    font-weight: 700;
    border-radius: 10px;
}

/* ── expander 스타일 ── */
details summary {
    font-size: 15px !important;
    font-weight: 600;
    padding: 12px 0 !important;
}

/* ── iOS 입력 zoom 방지 ── */
input, select, textarea {
    font-size: 16px !important;
}

/* ── 탭 버튼 ── */
button[data-baseweb="tab"] {
    font-size: 14px !important;
    padding: 10px 8px !important;
    min-height: 44px !important;
}

/* ── 데이터프레임 폰트 ── */
div[data-testid="stDataFrame"] {
    font-size: 13px;
}

/* ── 큰 화면에서는 카드 값 더 크게 ── */
@media (min-width: 480px) {
    .metric-value  { font-size: 22px; }
    .metric-label  { font-size: 12px; }
    .metric-delta  { font-size: 12px; }
    .section-header { font-size: 17px; }
    .app-title     { font-size: 26px; }
}
</style>
""", unsafe_allow_html=True)

# ── 상수 ───────────────────────────────────────────────────────────────────────
CURRENCY_PAIRS = {
    "USD/KRW 달러/원":    "USDKRW=X",
    "EUR/KRW 유로/원":    "EURKRW=X",
    "JPY/KRW 엔/원":      "JPYKRW=X",
    "CNY/KRW 위안/원":    "CNYKRW=X",
    "GBP/KRW 파운드/원":  "GBPKRW=X",
    "EUR/USD 유로/달러":  "EURUSD=X",
    "GBP/USD 파운드/달러":"GBPUSD=X",
    "USD/JPY 달러/엔":    "USDJPY=X",
}

PERIOD_OPTIONS = {
    "1개월":   30,
    "3개월":   90,
    "6개월":  180,
    "1년":    365,
    "2년":    730,
}

FORECAST_DAYS_OPTIONS = [30, 60, 90]

# ── 설정 영속성 ────────────────────────────────────────────────────────────────
SETTINGS_FILE = Path(__file__).parent / ".settings.json"

DEFAULT_SETTINGS = {
    "pair_label":    "USD/KRW 달러/원",
    "period_label":  "1년",
    "forecast_days": 30,
    "model_choice":  ["ARIMA", "선형회귀(LR)"],
    "show_ma":       True,
    "show_bb":       True,
    "show_vol":      True,
    "forecast_start": None,   # None → 오늘 기준 자동 계산
}

def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def _save_settings():
    """session_state → .settings.json 자동 저장"""
    data = {
        "pair_label":    st.session_state.get("pair_label",    DEFAULT_SETTINGS["pair_label"]),
        "period_label":  st.session_state.get("period_label",  DEFAULT_SETTINGS["period_label"]),
        "forecast_days": st.session_state.get("forecast_days", DEFAULT_SETTINGS["forecast_days"]),
        "model_choice":  st.session_state.get("model_choice",  DEFAULT_SETTINGS["model_choice"]),
        "show_ma":       st.session_state.get("show_ma",       DEFAULT_SETTINGS["show_ma"]),
        "show_bb":       st.session_state.get("show_bb",       DEFAULT_SETTINGS["show_bb"]),
        "show_vol":      st.session_state.get("show_vol",      DEFAULT_SETTINGS["show_vol"]),
        "forecast_start": st.session_state["forecast_start"].isoformat()
                          if st.session_state.get("forecast_start") else None,
    }
    try:
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# 앱 첫 실행 시 저장된 설정을 session_state에 로드
if "settings_loaded" not in st.session_state:
    _s = _load_settings()
    st.session_state["pair_label"]    = _s["pair_label"]    if _s["pair_label"]   in CURRENCY_PAIRS else DEFAULT_SETTINGS["pair_label"]
    st.session_state["period_label"]  = _s["period_label"]  if _s["period_label"] in PERIOD_OPTIONS  else DEFAULT_SETTINGS["period_label"]
    st.session_state["forecast_days"] = _s["forecast_days"] if _s["forecast_days"] in FORECAST_DAYS_OPTIONS else DEFAULT_SETTINGS["forecast_days"]
    st.session_state["model_choice"]  = _s["model_choice"]
    st.session_state["show_ma"]       = _s["show_ma"]
    st.session_state["show_bb"]       = _s["show_bb"]
    st.session_state["show_vol"]      = _s["show_vol"]
    # 예측 시작일: 저장된 날짜가 있으면 복원, 없으면 None (expander에서 자동 계산)
    if _s["forecast_start"]:
        try:
            st.session_state["forecast_start"] = datetime.fromisoformat(_s["forecast_start"]).date()
        except Exception:
            st.session_state["forecast_start"] = None
    else:
        st.session_state["forecast_start"] = None
    st.session_state["settings_loaded"] = True

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(ticker: str, start: str, end: str):
    """(DataFrame, error_msg) 튜플 반환. 실패 시 빈 DataFrame + 오류 문자열."""
    last_err = ""
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                last_err = "Yahoo Finance 응답이 비어 있습니다 (rate limit 또는 잘못된 티커)."
                if attempt < 2:
                    time.sleep(1.5)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                df = close.rename("Close").to_frame()
            else:
                df = df[["Close"]].copy()
                df.columns = ["Close"]

            df.index = pd.to_datetime(df.index)
            return df.dropna(), ""

        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(1.5)

    return pd.DataFrame(), last_err


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MA5"]      = d["Close"].rolling(5).mean()
    d["MA20"]     = d["Close"].rolling(20).mean()
    d["MA60"]     = d["Close"].rolling(60).mean()
    d["Std20"]    = d["Close"].rolling(20).std()
    d["Upper"]    = d["MA20"] + 2 * d["Std20"]
    d["Lower"]    = d["MA20"] - 2 * d["Std20"]
    d["Return"]   = d["Close"].pct_change()
    d["Volatility"] = d["Return"].rolling(20).std() * np.sqrt(252) * 100
    return d


# ── 예측 모델 ─────────────────────────────────────────────────────────────────
def arima_forecast(series: pd.Series, steps: int, alpha: float = 0.1):
    try:
        result = ARIMA(series, order=(5, 1, 0)).fit()
        fc     = result.get_forecast(steps=steps)
        ci     = fc.conf_int(alpha=alpha)
        return fc.predicted_mean.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values
    except Exception as e:
        st.warning(f"ARIMA 오류: {e}")
        return None, None, None


def lr_forecast(series: pd.Series, steps: int, alpha: float = 0.1):
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1))
        window = min(30, len(scaled) // 2)

        X, y = [], []
        for i in range(window, len(scaled)):
            X.append(scaled[i - window:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)

        model = LinearRegression().fit(X, y)
        residual_std = np.std(y - model.predict(X))
        z = 1.645  # 90% CI

        last_w = scaled[-window:, 0]
        preds, lowers, uppers = [], [], []
        for step in range(1, steps + 1):
            p = model.predict(last_w.reshape(1, -1))[0]
            m = z * residual_std * np.sqrt(step)
            preds.append(p); lowers.append(p - m); uppers.append(p + m)
            last_w = np.append(last_w[1:], p)

        inv = lambda a: scaler.inverse_transform(np.array(a).reshape(-1, 1)).flatten()
        return inv(preds), inv(lowers), inv(uppers)
    except Exception as e:
        st.warning(f"선형회귀 오류: {e}")
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# ── 상단 메뉴 (사이드바 대체) ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="app-title">💱 환율 예측 대시보드</p>', unsafe_allow_html=True)

with st.expander("⚙️ 설정", expanded=False):
    pair_label = st.selectbox(
        "통화쌍", list(CURRENCY_PAIRS.keys()),
        key="pair_label", on_change=_save_settings,
    )
    ticker = CURRENCY_PAIRS[pair_label]

    period_label = st.selectbox(
        "과거 데이터 기간", list(PERIOD_OPTIONS.keys()),
        key="period_label", on_change=_save_settings,
    )
    history_days = PERIOD_OPTIONS[period_label]

    st.markdown("**예측 설정**")

    data_start = (datetime.today() - timedelta(days=history_days)).date()
    data_end   = datetime.today().date()
    fc_min     = data_start + timedelta(days=60)
    fc_default = data_end - timedelta(days=1)

    # 저장된 날짜가 현재 유효 범위 안에 있으면 복원, 아니면 기본값
    saved_fc = st.session_state.get("forecast_start")
    if saved_fc is None or not (fc_min <= saved_fc <= data_end):
        saved_fc = fc_default

    forecast_start = st.date_input(
        "예측 시작일",
        value=saved_fc,
        min_value=fc_min,
        max_value=data_end,
        key="forecast_start",
        on_change=_save_settings,
        help="이 날짜까지 학습 → 이후 예측. 과거 날짜 선택 시 실제값과 비교합니다.",
    )

    forecast_days = st.radio(
        "예측 기간", FORECAST_DAYS_OPTIONS,
        format_func=lambda x: f"{x}일",
        horizontal=True,
        key="forecast_days", on_change=_save_settings,
    )

    model_choice = st.multiselect(
        "예측 모델", ["ARIMA", "선형회귀(LR)"],
        key="model_choice", on_change=_save_settings,
    )

    st.markdown("**차트 옵션**")
    c1, c2, c3 = st.columns(3)
    show_ma  = c1.checkbox("이동평균",   key="show_ma",  on_change=_save_settings)
    show_bb  = c2.checkbox("볼린저밴드", key="show_bb",  on_change=_save_settings)
    show_vol = c3.checkbox("변동성",     key="show_vol", on_change=_save_settings)

    st.caption("✅ 설정은 자동 저장됩니다.")

st.markdown(f'<p class="app-subtitle">{pair_label} &nbsp;|&nbsp; 과거 {period_label} 데이터</p>',
            unsafe_allow_html=True)

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
end_date   = datetime.today()
start_date = end_date - timedelta(days=history_days)

with st.spinner("데이터 로딩 중..."):
    df_raw, load_err = load_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if df_raw.empty:
    st.error("데이터를 불러올 수 없습니다.")
    if load_err:
        st.code(load_err, language="")
    if st.button("🔄 다시 시도", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.stop()

df = add_features(df_raw)

# ── 지표 카드 (2×2 그리드) ────────────────────────────────────────────────────
latest   = float(df["Close"].iloc[-1])
prev     = float(df["Close"].iloc[-2]) if len(df) > 1 else latest
change   = latest - prev
change_p = (change / prev) * 100 if prev != 0 else 0
high_52  = float(df["Close"].tail(252).max())
low_52   = float(df["Close"].tail(252).min())
vol_now  = float(df["Volatility"].dropna().iloc[-1]) if not df["Volatility"].dropna().empty else 0.0

arrow = "▲" if change >= 0 else "▼"
dclr  = "#4ade80" if change >= 0 else "#f87171"

st.markdown(f"""
<div class="card-grid">
  <div class="metric-card">
    <div class="metric-label">현재 환율</div>
    <div class="metric-value">{latest:,.2f}</div>
    <div class="metric-delta" style="color:{dclr}">
      {arrow} {abs(change):.2f} ({abs(change_p):.2f}%)
    </div>
  </div>
  <div class="metric-card">
    <div class="metric-label">연환산 변동성</div>
    <div class="metric-value">{vol_now:.1f}%</div>
    <div class="metric-delta">최근 20일 기준</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">52주 최고</div>
    <div class="metric-value">{high_52:,.2f}</div>
    <div class="metric-delta">현재 대비 {((high_52-latest)/latest*100):+.2f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">52주 최저</div>
    <div class="metric-value">{low_52:,.2f}</div>
    <div class="metric-delta">현재 대비 {((low_52-latest)/latest*100):+.2f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── 탭 구성 ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 환율 추이", "🔮 예측", "📊 분석"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 : 환율 추이
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(df.index, df["Close"], color="#2563eb", linewidth=1.5, label="종가")

    if show_ma:
        if df["MA5"].notna().any():
            ax.plot(df.index, df["MA5"],  color="#f59e0b", linewidth=1,
                    linestyle="--", label="MA5")
        if df["MA20"].notna().any():
            ax.plot(df.index, df["MA20"], color="#10b981", linewidth=1,
                    linestyle="--", label="MA20")
        if df["MA60"].notna().any():
            ax.plot(df.index, df["MA60"], color="#ef4444", linewidth=1,
                    linestyle="--", label="MA60")

    if show_bb and df["Upper"].notna().any():
        ax.fill_between(df.index, df["Lower"], df["Upper"],
                        alpha=0.1, color="#8b5cf6")
        ax.plot(df.index, df["Upper"], color="#8b5cf6", linewidth=0.6,
                linestyle=":", label="볼린저밴드")
        ax.plot(df.index, df["Lower"], color="#8b5cf6", linewidth=0.6, linestyle=":")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    ax.set_ylabel("환율", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    if show_vol:
        st.markdown('<div class="section-header">📊 연환산 변동성</div>',
                    unsafe_allow_html=True)
        vol_data = df["Volatility"].dropna()
        fig2, ax2 = plt.subplots(figsize=(5.5, 2.2))
        ax2.fill_between(vol_data.index, vol_data, alpha=0.5, color="#f59e0b")
        ax2.plot(vol_data.index, vol_data, color="#d97706", linewidth=1)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        ax2.set_ylabel("변동성 (%)", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("#fffbeb")
        fig2.patch.set_facecolor("white")
        fig2.tight_layout(pad=0.8)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 : 예측
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not model_choice:
        st.info("설정에서 예측 모델을 1개 이상 선택해주세요.")
        st.stop()

    series       = df["Close"].dropna()
    fc_start_ts  = pd.Timestamp(forecast_start)
    train_series = series[series.index <= fc_start_ts]
    actual_in_fc = series[series.index > fc_start_ts]

    if len(train_series) < 60:
        st.warning("학습 데이터 부족. 예측 시작일을 뒤로 조정하거나 데이터 기간을 늘려주세요.")
        st.stop()

    future_dates  = pd.bdate_range(start=fc_start_ts + timedelta(days=1), periods=forecast_days)
    MODEL_COLORS  = {"ARIMA": "#ef4444", "선형회귀(LR)": "#10b981"}
    fc_results    = {}

    if "ARIMA" in model_choice:
        with st.spinner("ARIMA 학습 중..."):
            mn, lo, hi = arima_forecast(train_series, forecast_days)
        if mn is not None:
            fc_results["ARIMA"] = (mn, lo, hi)

    if "선형회귀(LR)" in model_choice:
        with st.spinner("선형회귀 학습 중..."):
            mn, lo, hi = lr_forecast(train_series, forecast_days)
        if mn is not None:
            fc_results["선형회귀(LR)"] = (mn, lo, hi)

    if not fc_results:
        st.error("모든 모델이 실패했습니다.")
        st.stop()

    # 예측 차트
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.8))

    context = train_series.tail(60)
    ax3.plot(context.index, context.values,
             color="#2563eb", linewidth=1.8, label="실제값", zorder=3)

    if not actual_in_fc.empty:
        actual_show = actual_in_fc[actual_in_fc.index <= future_dates[-1]]
        if not actual_show.empty:
            ax3.plot(actual_show.index, actual_show.values,
                     color="#2563eb", linewidth=1.8, linestyle=":",
                     label="실제(비교)", zorder=3)

    for model_name, (mean_vals, lo_vals, hi_vals) in fc_results.items():
        color  = MODEL_COLORS.get(model_name, "#6366f1")
        n      = min(len(mean_vals), len(future_dates))
        dates_n = future_dates[:n]

        ax3.fill_between(dates_n, lo_vals[:n], hi_vals[:n],
                         alpha=0.15, color=color)
        ax3.plot(dates_n, mean_vals[:n],
                 color=color, linewidth=2, linestyle="--",
                 label=f"{model_name}", zorder=4)
        ax3.annotate(
            f"{mean_vals[n-1]:,.0f}",
            xy=(dates_n[-1], mean_vals[n-1]),
            xytext=(5, 3), textcoords="offset points",
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
        )

    ax3.axvline(x=fc_start_ts, color="#6b7280", linestyle=":", linewidth=1.2)
    ax3.axvspan(fc_start_ts, future_dates[-1], alpha=0.04, color="#6b7280")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    ax3.set_ylabel("환율", fontsize=9)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#f8fafc")
    fig3.patch.set_facecolor("white")
    fig3.tight_layout(pad=0.8)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    # 예측 테이블
    st.markdown('<div class="section-header">📋 날짜별 예측값</div>',
                unsafe_allow_html=True)

    table_rows = []
    for i, d in enumerate(future_dates[:forecast_days]):
        row = {"날짜": d.strftime("%m/%d")}
        for mname, (mv, lv, hv) in fc_results.items():
            if i < len(mv):
                row[f"{mname}"] = f"{mv[i]:,.0f}"
                row[f"하한"]    = f"{lv[i]:,.0f}"
                row[f"상한"]    = f"{hv[i]:,.0f}"
        av = actual_in_fc[actual_in_fc.index.date == d.date()]
        if not av.empty:
            row["실제값"] = f"{float(av.iloc[0]):,.0f}"
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    st.dataframe(table_df, use_container_width=True,
                 height=min(380, 36 * min(forecast_days, 10) + 38))

    # 예측 정확도 (과거 날짜 선택 시)
    if not actual_in_fc.empty and len(actual_in_fc) >= 5:
        st.markdown('<div class="section-header">📐 예측 정확도</div>',
                    unsafe_allow_html=True)
        st.caption(f"예측 시작일 이후 실제 데이터 {len(actual_in_fc)}일치 비교")

        perf_rows = []
        for mname, (mv, _, _) in fc_results.items():
            act_l, pred_l = [], []
            for i, d in enumerate(future_dates[:len(mv)]):
                av = actual_in_fc[actual_in_fc.index.date == d.date()]
                if not av.empty:
                    act_l.append(float(av.iloc[0]))
                    pred_l.append(mv[i])
            if len(act_l) >= 2:
                a, p = np.array(act_l), np.array(pred_l)
                perf_rows.append({
                    "모델":  mname,
                    "일수":  len(a),
                    "MAE":   f"{mean_absolute_error(a,p):,.2f}",
                    "RMSE":  f"{np.sqrt(mean_squared_error(a,p)):,.2f}",
                    "MAPE":  f"{np.mean(np.abs((a-p)/a))*100:.2f}%",
                })
        if perf_rows:
            st.dataframe(pd.DataFrame(perf_rows), use_container_width=True,
                         hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 : 분석
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📉 일별 수익률 분포</div>',
                unsafe_allow_html=True)

    returns = df["Return"].dropna()
    fig4, ax4 = plt.subplots(figsize=(5.5, 2.8))
    ax4.hist(returns, bins=50, color="#2563eb", alpha=0.7,
             edgecolor="white", linewidth=0.4)
    ax4.axvline(returns.mean(), color="#ef4444", linewidth=1.5, linestyle="--",
                label=f"평균 {returns.mean():.4f}")
    ax4.axvline(0, color="#6b7280", linewidth=1, linestyle=":")
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    ax4.set_xlabel("수익률", fontsize=9)
    ax4.set_ylabel("빈도", fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#f8fafc")
    fig4.patch.set_facecolor("white")
    fig4.tight_layout(pad=0.8)
    st.pyplot(fig4, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-header">📊 기본 통계</div>',
                unsafe_allow_html=True)
    stat = df["Close"].describe()
    stat_df = pd.DataFrame({
        "항목": ["데이터 수", "평균", "표준편차", "최솟값",
                 "25%", "중앙값", "75%", "최댓값"],
        "값":   [f"{stat['count']:.0f}", f"{stat['mean']:,.2f}",
                 f"{stat['std']:,.2f}",  f"{stat['min']:,.2f}",
                 f"{stat['25%']:,.2f}",  f"{stat['50%']:,.2f}",
                 f"{stat['75%']:,.2f}",  f"{stat['max']:,.2f}"],
    })
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("⚠️ 예측 결과는 참고용이며 투자 조언이 아닙니다.")
