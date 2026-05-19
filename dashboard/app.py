"""
Streamlit dashboard for the Penalty Simulator (Pass 3).

Design blend:
  - Direction A: clean modular cards, neutral palette, generous whitespace
  - Direction D: real illustrated goal mouth for the heatmap, football accents
  - Direction C: editorial typography and section labels

IMPORTANT IMPLEMENTATION NOTE:
  All HTML strings sent to st.markdown(..., unsafe_allow_html=True) must have
  every `style="..."` attribute on a SINGLE LINE. Multi-line attributes confuse
  Streamlit's HTML parser and cause the markup to render as raw text. This is
  why styles are pre-built as flat strings before being interpolated.

Run:
    streamlit run dashboard/app.py
"""

import os
import streamlit as st
import plotly.graph_objects as go

from api_client import (
    APIError, health_check, list_takers, list_keepers, simulate,
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Penalty Simulator — WC 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ZONES = ['TL', 'TC', 'TR', 'BL', 'BC', 'BR']
ZONE_LABELS = {
    'TL': 'Top Left',
    'TC': 'Top Centre',
    'TR': 'Top Right',
    'BL': 'Bottom Left',
    'BC': 'Bottom Centre',
    'BR': 'Bottom Right',
}
OUTCOMES = ['GOAL', 'SAVED', 'POST', 'WIDE', 'OVER']
PHOTOS_DIR = "assets/photos"

COLORS = {
    'bg':          '#0a0a0a',
    'card':        '#161616',
    'card_alt':    '#1c1c1c',
    'border':      '#262626',
    'text':        '#e8e8e8',
    'text_muted':  '#8a8a8a',
    'text_dim':    '#5a5a5a',
    'accent':      '#22c55e',
    'accent_warn': '#f59e0b',
    'accent_bad':  '#ef4444',
    'goal_white':  '#e8e8e8',
}

RELIABILITY = {
    'high':   {'color': '#22c55e', 'label': 'High confidence',
               'tooltip': '10+ penalties in dataset — reliable predictions.'},
    'medium': {'color': '#f59e0b', 'label': 'Moderate confidence',
               'tooltip': '3-9 penalties in dataset — limited historical data.'},
    'low':    {'color': '#8a8a8a', 'label': 'Low confidence',
               'tooltip': '<3 penalties in dataset — predictions lean on population priors.'},
}

OUTCOME_COLORS = {
    'GOAL':  '#22c55e',
    'SAVED': '#ef4444',
    'POST':  '#f59e0b',
    'WIDE':  '#fb923c',
    'OVER':  '#f97316',
}


# ============================================================
# GLOBAL STYLES
# ============================================================

def inject_global_css():
    css = f"""
    <style>
    .stApp {{ background:{COLORS['bg']}; }}
    body, .stApp, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    .section-label {{
        color:{COLORS['text_dim']};
        font-size:0.7rem;
        font-weight:600;
        letter-spacing:0.12em;
        text-transform:uppercase;
        margin-bottom:0.4rem;
    }}
    .card {{
        background:{COLORS['card']};
        border:1px solid {COLORS['border']};
        border-radius:14px;
        padding:1.25rem;
    }}
    .stSelectbox > div > div, .stNumberInput > div > div {{
        background:{COLORS['card_alt']};
        border:1px solid {COLORS['border']};
        border-radius:8px;
    }}
    /* Pointer cursor on dropdowns so they feel clickable */
    .stSelectbox > div > div,
    .stSelectbox > div > div * {{
        cursor: pointer !important;
    }}
    .stButton > button {{
        background:{COLORS['accent']};
        color:#0a0a0a;
        font-weight:700;
        letter-spacing:0.03em;
        border-radius:10px;
        border:none;
        padding:0.7rem 1rem;
        transition:transform 0.1s ease;
    }}
    .stButton > button:hover {{
        background:#16a34a;
        transform:translateY(-1px);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

for k, v in [('simulation_result', None), ('history', []),
             ('selected_taker', None), ('selected_keeper', None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# HELPERS
# ============================================================

def silhouette_svg(size: int = 110) -> str:
    """Single-line SVG silhouette — same generic gray figure for everyone."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 110 110" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="55" cy="55" r="55" fill="#2a2a2a"/>'
        f'<circle cx="55" cy="44" r="17" fill="#5a5a5a"/>'
        f'<path d="M 22 105 Q 22 75 55 75 Q 88 75 88 105 Z" fill="#5a5a5a"/>'
        f'</svg>'
    )


def get_player_photo_html(player_id: int, size: int = 110) -> str:
    local_path = os.path.join(PHOTOS_DIR, f"{player_id}.jpg")
    if os.path.exists(local_path):
        img_style = f"border-radius:50%;object-fit:cover;height:{size}px;border:2px solid {COLORS['border']};"
        return f'<img src="{local_path}" width="{size}" style="{img_style}">'
    return silhouette_svg(size)


def reliability_badge_html(reliability: str) -> str:
    cfg = RELIABILITY.get(reliability, RELIABILITY['low'])
    style = (
        f"background:{cfg['color']}22;color:{cfg['color']};"
        f"padding:3px 10px;border-radius:6px;"
        f"font-size:0.7rem;font-weight:600;letter-spacing:0.03em;"
        f"border:1px solid {cfg['color']}55;cursor:help;"
    )
    return f'<span title="{cfg["tooltip"]}" style="{style}">{cfg["label"]}</span>'


@st.cache_data(ttl=300)
def cached_list_takers(limit: int = 300):
    return list_takers(limit=limit)


@st.cache_data(ttl=300)
def cached_list_keepers(limit: int = 300):
    return list_keepers(limit=limit)


def check_api():
    try:
        return health_check()
    except APIError as e:
        st.error(f"⚠ Cannot connect to the API. {e}")
        st.info("Run the API first:\n\n```bash\nuvicorn src.api.main:app --reload\n```")
        st.stop()


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    header_style = f"margin-bottom:2rem;padding-bottom:1.25rem;border-bottom:1px solid {COLORS['border']};"
    title_row_style = "display:flex;align-items:baseline;gap:0.6rem;"
    title_style = f"font-size:1.7rem;color:{COLORS['text']};margin:0;"
    kicker_style = f"color:{COLORS['accent']};font-weight:600;font-size:0.85rem;letter-spacing:0.05em;"
    sub_style = f"color:{COLORS['text_muted']};margin:0.4rem 0 0;font-size:0.9rem;"

    html = (
        f'<div style="{header_style}">'
        f'<div style="{title_row_style}">'
        f'<h1 style="{title_style}">Penalty Simulator</h1>'
        f'<span style="{kicker_style}">WORLD CUP 2026</span>'
        '</div>'
        f'<p style="{sub_style}">Predict any penalty matchup using ML models trained on 1,000+ historical penalties.</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_player_card(player: dict, role: str):
    if role == 'taker':
        player_id = player['taker_id']
        name = player['taker_name']
        sample_size = player['total_penalties']
    else:
        player_id = player['keeper_id']
        name = player['keeper_name']
        sample_size = player['total_penalties_faced']

    reliability = player.get('reliability', 'low')
    role_label = "TAKER" if role == 'taker' else "KEEPER"
    tooltip = "Penalties found in our training data (StatsBomb open data). Doesn't include matches outside that dataset."

    photo_html = get_player_photo_html(player_id, size=110)
    badge_html = reliability_badge_html(reliability)

    label_style = f"color:{COLORS['text_dim']};font-size:0.65rem;font-weight:600;letter-spacing:0.12em;margin-bottom:0.6rem;"
    name_style = f"margin-top:0.75rem;font-weight:600;font-size:1.05rem;color:{COLORS['text']};line-height:1.2;"
    pens_style = f"margin-top:0.85rem;color:{COLORS['text_muted']};font-size:0.85rem;"
    pens_num_style = f"font-weight:700;color:{COLORS['text']};"

    html = (
        '<div class="card" style="text-align:center;height:100%;">'
        f'<div style="{label_style}">{role_label}</div>'
        f'{photo_html}'
        f'<div style="{name_style}">{name}</div>'
        f'<div style="{pens_style}" title="{tooltip}">'
        f'<span style="{pens_num_style}">{sample_size}</span> penalties in dataset'
        '</div>'
        f'<div style="margin-top:0.7rem;">{badge_html}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_goal_probability_gauge(p_goal: float):
    pct = round(p_goal * 100, 1)
    color = COLORS['accent'] if pct >= 75 else COLORS['accent_warn'] if pct >= 50 else COLORS['accent_bad']

    card_style = (
        f"background:linear-gradient(135deg,{COLORS['card']} 0%,{COLORS['card_alt']} 100%);"
        f"padding:1.75rem 1rem;text-align:center;"
    )
    label_style = "margin-bottom:0.4rem;"
    number_style = (
        f"font-size:4.2rem;font-weight:800;color:{color};line-height:1;"
        f"font-variant-numeric:tabular-nums;letter-spacing:-0.02em;"
    )
    pct_sign_style = f"font-size:2.2rem;font-weight:600;color:{COLORS['text_muted']};"
    bar_outer_style = (
        f"margin-top:1rem;background:{COLORS['bg']};border-radius:8px;"
        f"height:10px;overflow:hidden;border:1px solid {COLORS['border']};"
    )
    bar_inner_style = f"width:{pct}%;background:{color};height:100%;transition:width 0.4s ease;"

    html = (
        f'<div class="card" style="{card_style}">'
        f'<div class="section-label" style="{label_style}">Probability of Goal</div>'
        f'<div style="{number_style}">{pct}<span style="{pct_sign_style}">%</span></div>'
        f'<div style="{bar_outer_style}"><div style="{bar_inner_style}"></div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_goal_mouth(zone_probs: dict):
    max_p = max(zone_probs.values()) if zone_probs else 1
    if max_p == 0:
        max_p = 1

    rows = [['TL', 'TC', 'TR'], ['BL', 'BC', 'BR']]
    cell_w = (520 - 80) / 3
    cell_h = (260 - 40) / 2

    cells_svg = ""
    for r_idx, row in enumerate(rows):
        for c_idx, zone in enumerate(row):
            x = 80 + c_idx * cell_w
            y = 40 + r_idx * cell_h
            prob = zone_probs.get(zone, 0)
            intensity = prob / max_p
            r = int(255 - intensity * 30)
            g = int(220 - intensity * 180)
            b = int(80 - intensity * 60)
            a = 0.25 + intensity * 0.65
            pct = round(prob * 100, 1)
            cells_svg += (
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="rgba({r},{g},{b},{a})" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'
                f'<text x="{x + cell_w/2}" y="{y + cell_h/2 - 6}" text-anchor="middle" '
                f'fill="#ffffff" opacity="0.6" font-family="Inter, sans-serif" '
                f'font-size="11" font-weight="600">{zone}</text>'
                f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 18}" text-anchor="middle" '
                f'fill="#ffffff" font-family="Inter, sans-serif" font-size="22" font-weight="700">{pct}%</text>'
            )

    svg = (
        '<svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">'
        f'<rect x="0" y="0" width="600" height="320" fill="{COLORS["card"]}"/>'
        f'<line x1="0" y1="278" x2="600" y2="278" stroke="{COLORS["border"]}" stroke-width="1"/>'
        f'<circle cx="300" cy="295" r="3" fill="{COLORS["text_dim"]}"/>'
        '<defs>'
        '<pattern id="net" patternUnits="userSpaceOnUse" width="14" height="14">'
        '<path d="M 0 14 L 14 0" stroke="rgba(255,255,255,0.06)" stroke-width="0.8"/>'
        '<path d="M 0 0 L 14 14" stroke="rgba(255,255,255,0.06)" stroke-width="0.8"/>'
        '</pattern>'
        '</defs>'
        '<rect x="80" y="40" width="440" height="220" fill="url(#net)"/>'
        f'{cells_svg}'
        f'<line x1="78" y1="40" x2="78" y2="262" stroke="{COLORS["goal_white"]}" stroke-width="6" stroke-linecap="round"/>'
        f'<line x1="522" y1="40" x2="522" y2="262" stroke="{COLORS["goal_white"]}" stroke-width="6" stroke-linecap="round"/>'
        f'<line x1="75" y1="40" x2="525" y2="40" stroke="{COLORS["goal_white"]}" stroke-width="6" stroke-linecap="round"/>'
        '</svg>'
    )

    # Build the legend: 6 zone codes mapped to their full names, in a horizontal flow
    legend_item_style = (
        f"display:inline-flex;align-items:center;gap:0.35rem;"
        f"color:{COLORS['text_muted']};font-size:0.72rem;"
        f"margin:0 0.6rem;"
    )
    code_style = (
        f"display:inline-block;min-width:22px;padding:1px 6px;"
        f"background:{COLORS['card_alt']};border:1px solid {COLORS['border']};"
        f"border-radius:4px;font-size:0.65rem;font-weight:700;"
        f"color:{COLORS['text']};text-align:center;letter-spacing:0.02em;"
    )

    legend_items = ""
    for z in ZONES:
        legend_items += (
            f'<span style="{legend_item_style}">'
            f'<span style="{code_style}">{z}</span>'
            f'{ZONE_LABELS[z]}'
            f'</span>'
        )

    legend_wrap_style = (
        f"display:flex;flex-wrap:wrap;justify-content:center;"
        f"gap:0.25rem 0;padding:0.85rem 0.25rem 0.25rem;"
        f"border-top:1px solid {COLORS['border']};margin-top:0.75rem;"
    )
    caption_style = (
        f"text-align:center;color:{COLORS['text_dim']};"
        f"font-size:0.7rem;margin-top:0.4rem;"
    )

    html = (
        '<div class="card" style="padding:1.25rem 1rem 0.75rem;">'
        '<div class="section-label">Predicted Shot Location</div>'
        f'<div style="margin-top:0.5rem;">{svg}</div>'
        f'<div style="{legend_wrap_style}">{legend_items}</div>'
        f'<div style="{caption_style}">Viewed from taker\'s perspective · darker = more likely</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_outcome_breakdown(outcome_probs: dict):
    sorted_outcomes = sorted(outcome_probs.items(), key=lambda x: x[1])
    labels = [oc for oc, _ in sorted_outcomes]
    values = [round(p * 100, 1) for _, p in sorted_outcomes]
    colors = [OUTCOME_COLORS.get(oc, '#7f8c8d') for oc in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=13, family='Inter'),
        hovertemplate='<b>%{y}</b><br>%{x}%<extra></extra>',
    ))
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        margin=dict(l=10, r=50, t=10, b=10),
        height=220,
        xaxis=dict(range=[0, max(values) * 1.25 if values else 100],
                   showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(color=COLORS['text'], size=13, family='Inter'), showgrid=False),
        showlegend=False,
    )

    label_html = '<div class="card" style="padding:1.25rem;"><div class="section-label">Outcome Distribution</div></div>'
    st.markdown(label_html, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)


def render_shootout_context(pressure: str, kick_num: int):
    descriptions = {
        'MUST_SCORE': ('MUST SCORE', COLORS['accent_bad'],
                       f'Kick #{kick_num} — Missing this kick eliminates the team. '
                       'The highest-pressure scenario in football.'),
        'CAN_WIN':    ('CAN WIN IT', COLORS['accent'],
                       f'Kick #{kick_num} — Scoring this kick wins the shootout. '
                       'The most favourable scenario for the taker.'),
        'STANDARD':   ('STANDARD KICK', '#3b82f6',
                       f'Kick #{kick_num} — No immediate elimination or win at stake.'),
    }
    title, color, desc = descriptions.get(pressure, ('', COLORS['text_dim'], ''))
    if not title:
        return

    card_style = f"border-left:4px solid {color};margin:0 0 1rem;"
    title_style = f"color:{color};font-weight:700;font-size:0.78rem;letter-spacing:0.12em;"
    desc_style = f"color:{COLORS['text_muted']};font-size:0.85rem;margin-top:0.35rem;"

    html = (
        f'<div class="card" style="{card_style}">'
        f'<div style="{title_style}">{title}</div>'
        f'<div style="{desc_style}">{desc}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    inject_global_css()
    api_info = check_api()
    render_header()

    # ----- Player selection -----
    col_taker, col_keeper = st.columns(2)

    with col_taker:
        st.markdown('<div class="section-label">Taker</div>', unsafe_allow_html=True)
        takers = cached_list_takers(limit=300)
        taker_options = {t['taker_name']: t for t in takers}
        names = list(taker_options.keys())
        default_idx = next(
            (i for i, n in enumerate(names) if taker_options[n]['reliability'] == 'high'), 0
        )
        selected = st.selectbox("Choose a taker", options=names, index=default_idx,
                                key='taker_select', label_visibility='collapsed')
        selected_taker = taker_options[selected]
        st.session_state.selected_taker = selected_taker

    with col_keeper:
        st.markdown('<div class="section-label">Keeper</div>', unsafe_allow_html=True)
        keepers = cached_list_keepers(limit=300)
        keeper_options = {k['keeper_name']: k for k in keepers}
        names = list(keeper_options.keys())
        default_idx = next(
            (i for i, n in enumerate(names) if keeper_options[n]['reliability'] == 'high'), 0
        )
        selected = st.selectbox("Choose a keeper", options=names, index=default_idx,
                                key='keeper_select', label_visibility='collapsed')
        selected_keeper = keeper_options[selected]
        st.session_state.selected_keeper = selected_keeper

    # ----- Context -----
    st.markdown('<div class="section-label" style="margin-top:1.25rem;">Match Context</div>',
                unsafe_allow_html=True)
    ctx1, ctx2, ctx3, ctx4 = st.columns([1, 1.2, 1, 1])

    pressure, game_state, kick_num, minute = 'STANDARD', 'LEVEL', 3, 60
    with ctx1:
        is_shootout = st.toggle("Shootout", value=False)
    with ctx2:
        if is_shootout:
            pressure = st.selectbox("Pressure", options=['STANDARD', 'CAN_WIN', 'MUST_SCORE'], index=0)
        else:
            game_state = st.selectbox("Score state", options=['LEVEL', 'LEADING', 'TRAILING'], index=0)
    with ctx3:
        if is_shootout:
            kick_num = st.number_input("Kick #", min_value=1, max_value=11, value=3)
        else:
            minute = st.number_input("Minute", min_value=1, max_value=120, value=60)
    with ctx4:
        n_simulations = st.selectbox("Simulations", options=[1000, 5000, 10000, 25000], index=2)

    simulate_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if simulate_btn:
        payload = {
            'taker_id': int(selected_taker['taker_id']),
            'keeper_id': int(selected_keeper['keeper_id']),
            'is_shootout': bool(is_shootout),
            'n_simulations': int(n_simulations),
        }
        if is_shootout:
            payload['shootout_pressure'] = pressure
            payload['shootout_kick_num'] = int(kick_num)
            payload['minute'] = 120
        else:
            payload['game_state'] = game_state
            payload['minute'] = int(minute)

        with st.spinner("Running simulation…"):
            try:
                result = simulate(payload)
                st.session_state.simulation_result = result
                st.session_state.history.insert(0, {
                    'taker': selected_taker['taker_name'],
                    'keeper': selected_keeper['keeper_name'],
                    'p_goal': result['p_goal'],
                    'shootout': is_shootout,
                    'pressure': pressure if is_shootout else None,
                })
                st.session_state.history = st.session_state.history[:10]
            except APIError as e:
                st.error(f"Simulation failed: {e}")

    # ----- Results -----
    result = st.session_state.simulation_result
    if result is None:
        empty_style = (
            f"border-style:dashed;padding:3rem 1rem;text-align:center;"
            f"margin-top:1.5rem;color:{COLORS['text_dim']};"
        )
        ball_style = "font-size:2.2rem;margin-bottom:0.5rem;"
        text_style = f"color:{COLORS['text_muted']};"
        empty_html = (
            f'<div class="card" style="{empty_style}">'
            f'<div style="{ball_style}">⚽</div>'
            f'<div style="{text_style}">Pick a taker and keeper, then run a simulation.</div>'
            '</div>'
        )
        st.markdown(empty_html, unsafe_allow_html=True)
        return

    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    if is_shootout:
        render_shootout_context(pressure, kick_num)

    render_goal_probability_gauge(result['p_goal'])

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2.2, 1])
    with c1:
        render_player_card(selected_taker, 'taker')
    with c2:
        render_goal_mouth(result['zone_probs'])
    with c3:
        render_player_card(selected_keeper, 'keeper')

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    render_outcome_breakdown(result['outcome_probs'])

    if st.session_state.history:
        with st.expander("Recent simulations", expanded=False):
            for h in st.session_state.history:
                icon = "🥅" if h['shootout'] else "⚽"
                ctx = f" · {h['pressure']}" if h.get('pressure') else ""
                pct = h['p_goal'] * 100
                line = (
                    f"{icon} **{h['taker']}** vs **{h['keeper']}**{ctx} — "
                    f"<span style='color:{COLORS['accent']};font-weight:600;'>"
                    f"{pct:.1f}%</span>"
                )
                st.markdown(line, unsafe_allow_html=True)

    footer_style = (
        f"margin-top:3rem;padding-top:1rem;border-top:1px solid {COLORS['border']};"
        f"color:{COLORS['text_dim']};font-size:0.72rem;text-align:center;"
    )
    footer_html = (
        f'<div style="{footer_style}">'
        f'Trained on {api_info.get("n_takers", "?")} takers and '
        f'{api_info.get("n_keepers", "?")} keepers from StatsBomb open data'
        '</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
