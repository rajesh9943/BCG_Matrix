import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

st.set_page_config(
    page_title="BCG Matrix Stock Classifier",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #F8F8F6; }
    .hero-title {
        font-size: 2.2rem; font-weight: 600;
        letter-spacing: -0.03em; color: #1A1A18; margin-bottom: 0.2rem;
    }
    .hero-sub { font-size: 0.95rem; color: #888880; margin-bottom: 1.5rem; }
    .metric-box {
        background: white; border: 1px solid rgba(0,0,0,0.08);
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .stButton > button {
        background: #1A1A18; color: white; border: none;
        border-radius: 8px; font-family: 'DM Sans', sans-serif;
        font-weight: 500; padding: 0.5rem 1.5rem;
        width: 100%; font-size: 0.95rem;
    }
    .stButton > button:hover { background: #333330; color: white; }
    div[data-testid="stNumberInput"] input {
        font-family: 'DM Mono', monospace; font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "stocks" not in st.session_state:
    st.session_state.stocks = []

# ── Helpers ───────────────────────────────────────────────────────────────────
def classify(growth: float, share: float) -> str:
    """
    BCG Matrix classification:
    - Star:          High growth (>10%) + High share (>=1.0)
    - Cash Cow:      Low growth (<=10%) + High share (>=1.0)
    - Question Mark: High growth (>10%) + Low share (<1.0)
    - Dog:           Low growth (<=10%) + Low share (<1.0)
    """
    if growth > 10 and share >= 1.0:  return "Star"
    if growth <= 10 and share >= 1.0: return "Cash Cow"
    if growth > 10 and share < 1.0:   return "Question Mark"
    return "Dog"

CATEGORY_META = {
    "Star":          {"color": "#E8920A", "label_color": "#A06010",
                      "desc": "High growth, high market share. Strong competitor in a growing market. Invest to maintain leadership."},
    "Cash Cow":      {"color": "#4A9A1A", "label_color": "#3B6D11",
                      "desc": "Low growth, high market share. Mature market leader generating steady cash. Fund other units from profits."},
    "Question Mark": {"color": "#2878CC", "label_color": "#185FA5",
                      "desc": "High growth, low market share. Uncertain future — needs strategic decision to invest heavily or divest."},
    "Dog":           {"color": "#777770", "label_color": "#5F5E5A",
                      "desc": "Low growth, low market share. Weak position in a stagnant market. Consider phasing out or restructuring."},
}

# BCG Standard quadrant colors:
#   star  → top-LEFT    (High Share + High Growth)
#   qmark → top-RIGHT   (Low Share  + High Growth)
#   cash  → bottom-LEFT (High Share + Low Growth)
#   dog   → bottom-RIGHT(Low Share  + Low Growth)
QUAD_COLORS = {
    "star":  "#C8D0F0",  # soft lavender blue
    "qmark": "#F8DDB0",  # soft mint green
    "cash":  "#F5C8C8",  # soft peach/orange
    "dog":   "#B8E8C8",  # soft pink/rose
}


def draw_matrix(stocks: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    MID_X = 1.0   # relative share threshold — >= 1.0 is market leader
    MID_Y = 10.0  # growth threshold         — > 10% is high growth

    # ── Dynamic axis limits based on actual data ──────────────────────────────
    all_x = [s["share"]  for s in stocks] if stocks else [4.0]
    all_y = [s["growth"] for s in stocks] if stocks else [40.0]

    raw_x_max = max(all_x)
    raw_y_max = max(all_y)
    raw_y_min = min(all_y)

    x_pad = max(raw_x_max * 0.20, 0.6)
    y_pad = max((raw_y_max - raw_y_min) * 0.20, 6)

    X_MIN = 0.0
    X_MAX = max(4.0,  raw_x_max + x_pad)
    Y_MIN = min(-12,  raw_y_min - y_pad)
    Y_MAX = max(42,   raw_y_max + y_pad)

    x_range = X_MAX - X_MIN
    y_range = Y_MAX - Y_MIN

    # Smart tick spacing
    if x_range <= 5:    x_tick = 0.5
    elif x_range <= 12: x_tick = 1.0
    elif x_range <= 25: x_tick = 2.0
    else:               x_tick = round(x_range / 8 / 5) * 5

    if y_range <= 80:   y_tick = 10
    elif y_range <= 160:y_tick = 20
    else:               y_tick = round(y_range / 8 / 10) * 10

    # Quadrant label positions
    ql_xl = X_MIN + (MID_X - X_MIN) * 0.5    # centre of LEFT half  (High Share)
    ql_xr = MID_X + (X_MAX - MID_X) * 0.5    # centre of RIGHT half (Low Share)
    ql_yt = MID_Y + (Y_MAX - MID_Y) * 0.75   # upper portion
    ql_yb = Y_MIN + (MID_Y - Y_MIN) * 0.18   # lower portion

    # ── Quadrant fills (BCG Standard) ─────────────────────────────────────────
    #
    #   X-axis: LEFT side [X_MIN, MID_X] = share >= 1.0 → HIGH share (market leaders)
    #           RIGHT side [MID_X, X_MAX] = share <  1.0 → LOW share  (challengers)
    #
    #   ┌────────────────┬────────────────┐
    #   │  STARS ⭐      │ QUESTION MARKS │  ← High Growth (> 10%)
    #   │ (High Share)   │  (Low Share)   │
    #   ├────────────────┼────────────────┤
    #   │  CASH COWS 🐄  │    DOGS 🐕     │  ← Low Growth (<= 10%)
    #   │ (High Share)   │  (Low Share)   │
    #   └────────────────┴────────────────┘
    #     Left (≥1.0×)      Right (<1.0×)

    # ✅ top-left = Stars (High Growth + High Share)
    ax.fill_between([X_MIN, MID_X], [MID_Y, MID_Y], [Y_MAX, Y_MAX],
                    color=QUAD_COLORS["star"],  alpha=1.0, zorder=0)

    # ✅ top-right = Question Marks (High Growth + Low Share)
    ax.fill_between([MID_X, X_MAX], [MID_Y, MID_Y], [Y_MAX, Y_MAX],
                    color=QUAD_COLORS["qmark"], alpha=1.0, zorder=0)

    # ✅ bottom-left = Cash Cows (Low Growth + High Share)
    ax.fill_between([X_MIN, MID_X], [Y_MIN, Y_MIN], [MID_Y, MID_Y],
                    color=QUAD_COLORS["cash"],  alpha=1.0, zorder=0)

    # ✅ bottom-right = Dogs (Low Growth + Low Share)
    ax.fill_between([MID_X, X_MAX], [Y_MIN, Y_MIN], [MID_Y, MID_Y],
                    color=QUAD_COLORS["dog"],   alpha=1.0, zorder=0)

    # ── Dividers ──────────────────────────────────────────────────────────────
    ax.axhline(MID_Y, color="#AAAAAA", linewidth=1.0, linestyle="--", zorder=2)
    ax.axvline(MID_X, color="#AAAAAA", linewidth=1.0, linestyle="--", zorder=2)

    # ── Quadrant labels (corrected positions) ─────────────────────────────────
    for (x, y, txt, col) in [
        (ql_xl, ql_yt, "QUESTION\nMARK", "#1d3557"),   # ✅ top-right
        (ql_xr, ql_yt, "STAR","#1d3557"),   # ✅  top-left
        (ql_xl, ql_yb, "DOG","#1d3557"),   # ✅ bottom-right
        (ql_xr, ql_yb, "CASH COW","#1d3557"),   # ✅ bottom-left
    ]:
        ax.text(x, y, txt, fontsize=9, color=col, fontweight="700",
                ha="center", va="center", alpha=0.55, fontfamily="monospace")

    # ── Plot dots ─────────────────────────────────────────────────────────────
    lbl_offset = y_range * 0.046
    for s in stocks:
        color = CATEGORY_META[s["category"]]["color"]
        x, y = s["share"], s["growth"]
        ax.scatter(x, y, s=300, color=color, zorder=5,
                   edgecolors="white", linewidths=2.0)
        ax.text(x, y - lbl_offset, s["name"], fontsize=7.0,
                ha="center", va="top", color="#444440", zorder=6)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    # ✅ Clarified X-axis label showing share direction
    ax.set_xlabel("Relative Market Share  (←  Low share |  High share →)", fontsize=9,
                  color="#666660", labelpad=10)
    ax.set_ylabel("Market Growth Rate (%)  →", fontsize=9,
                  color="#666660", labelpad=10)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick))
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#DDDDDA")
        spine.set_linewidth(0.8)

    # ── Threshold annotations ─────────────────────────────────────────────────
    ax.annotate("", xy=(MID_X, Y_MIN + 1),
                fontsize=7, color="#AAAAAA", ha="center",
                fontfamily="monospace")
    ax.annotate("", xy=(X_MAX - 0.05, MID_Y + 0.8),
                fontsize=7, color="#AAAAAA", ha="right",
                fontfamily="monospace")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=CATEGORY_META[c]["color"], label=c, linewidth=0)
        for c in CATEGORY_META
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.95, edgecolor="#DDDDDA", facecolor="white",
              title="Category", title_fontsize=8)

    fig.tight_layout()
    return fig


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">BCG Matrix Stock Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Classify stocks by market growth rate and relative market share.</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.6], gap="large")

# ── Left panel ────────────────────────────────────────────────────────────────
with left:
    st.markdown("#### Add a stock")

    name = st.text_input(
        "Company name",
        placeholder="e.g. Sun Pharma",
        help="Enter the company or stock name."
    )

    growth = st.slider(
        "Market growth rate (%)",
        min_value=-10.0, max_value=200.0, value=10.0, step=1.0,
        help="Use 3-Year CAGR from Compounded Sales Growth. Threshold: > 10% = High growth."
    )

    share = st.number_input(
        "Relative market share",
        min_value=0.01, max_value=500.0, value=1.0, step=0.01,
        format="%.2f",
        help="Company Revenue ÷ Largest Competitor Revenue. Type exact value e.g. 1.67. ≥ 1.0 = market leader."
    )

    # Live preview card
    preview_cat = classify(growth, share)
    meta = CATEGORY_META[preview_cat]
    st.markdown(f"""
    <div style="background:white;border:1px solid rgba(0,0,0,0.08);border-radius:10px;
                padding:0.75rem 1rem;margin:0.6rem 0 1rem;">
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.09em;
                    text-transform:uppercase;color:{meta['label_color']};margin-bottom:3px;">
            Preview category
        </div>
        <div style="font-size:1rem;font-weight:600;color:#1A1A18;">{preview_cat}</div>
        <div style="font-size:0.8rem;color:#666660;line-height:1.5;margin-top:4px;">
            {meta['desc']}
        </div>
        <div style="font-size:0.72rem;color:#AAAAAA;margin-top:6px;font-family:monospace;">
            Growth: {growth:+.1f}% &nbsp;|&nbsp; Rel. Share: {share:.2f}&times;
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Add to matrix"):
            if not name.strip():
                st.warning("Please enter a company name.")
            else:
                existing_names = [s["name"].lower() for s in st.session_state.stocks]
                if name.strip().lower() in existing_names:
                    st.error(f"**{name.strip()}** is already in the matrix. Use a different name or clear the existing entry.")
                else:
                    st.session_state.stocks.append({
                        "name":     name.strip(),
                        "growth":   round(growth, 2),
                        "share":    round(share, 2),
                        "category": classify(growth, share),
                    })
                    st.success(f"**{name.strip()}** → **{classify(growth, share)}**")
                    st.rerun()
    with col_b:
        if st.button("Clear all"):
            st.session_state.stocks = []
            st.rerun()

    st.markdown("---")
    st.markdown("##### Quick reference")
    st.markdown("""
    <div style="font-size:0.8rem;color:#666660;line-height:2.0;">
        <b>Market growth</b> &rarr; 3-Year Compounded Sales Growth CAGR<br>
        <b>Relative share</b> &rarr; Company Revenue &divide; Largest Competitor Revenue<br>
        <b>Threshold</b> &rarr; Growth &gt; 10% = High &nbsp;|&nbsp; Share &ge; 1.0&times; = Leader<br><br>
        <b>BCG Quadrants:</b><br>
        ⭐ <b>Stars</b> &rarr; High Growth + High Share (top-right)<br>
        ❓ <b>Question Marks</b> &rarr; High Growth + Low Share (top-left)<br>
        🐄 <b>Cash Cows</b> &rarr; Low Growth + High Share (bottom-right)<br>
        🐕 <b>Dogs</b> &rarr; Low Growth + Low Share (bottom-left)
    </div>
    """, unsafe_allow_html=True)


# ── Right panel ───────────────────────────────────────────────────────────────
with right:
    st.markdown("#### BCG matrix")

    if st.session_state.stocks:
        fig = draw_matrix(st.session_state.stocks)
        st.pyplot(fig, use_container_width=True)

        st.markdown("#### Portfolio summary")
        df_s   = pd.DataFrame(st.session_state.stocks)
        counts = df_s["category"].value_counts()
        cols   = st.columns(4)
        for i, (cat, color) in enumerate(zip(
            ["Star", "Cash Cow", "Question Mark", "Dog"],
            ["#E8920A", "#4A9A1A", "#2878CC", "#777770"]
        )):
            with cols[i]:
                count = counts.get(cat, 0)
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:1.6rem;font-weight:600;color:{color};">{count}</div>
                    <div style="font-size:0.65rem;color:#888880;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.07em;margin-top:2px;">{cat}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:420px;display:flex;align-items:center;justify-content:center;
                    background:white;border:1px solid rgba(0,0,0,0.07);border-radius:12px;
                    color:#AAAAAA;font-size:0.9rem;">
            Add a stock on the left to plot it here.
        </div>
        """, unsafe_allow_html=True)


# ── Stock table ───────────────────────────────────────────────────────────────
if st.session_state.stocks:
    st.markdown("---")
    st.markdown("#### Added stocks")

    df         = pd.DataFrame(st.session_state.stocks)
    df_display = df.copy()
    df_display.columns = ["Company", "Growth (%)", "Rel. Share (×)", "Category"]
    df_display["Growth (%)"]     = df_display["Growth (%)"].map(lambda x: f"{x:+.1f}%")
    df_display["Rel. Share (×)"] = df_display["Rel. Share (×)"].map(lambda x: f"{x:.2f}×")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="bcg_stocks.csv",
        mime="text/csv",
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.75rem;color:#BBBBBA;text-align:center;padding:0.4rem 0;">
    BCG Matrix &nbsp;&middot;&nbsp; Growth &gt; 10% = High &nbsp;&middot;&nbsp;
    Relative share &ge; 1.0&times; = Market leader &nbsp;&middot;&nbsp;
    Stars &amp; Cash Cows → Left (High Share) | Question Marks &amp; Dogs → Right (Low Share)
</div>
""", unsafe_allow_html=True)
