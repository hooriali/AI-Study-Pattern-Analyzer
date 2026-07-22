import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

CHARTS_DIR = os.path.join('static', 'charts')

# ── Shared style applied to every chart ──────────────────────────────────────
PALETTE = {
    'dark':    '#212529',
    'mid':     '#495057',
    'light':   '#adb5bd',
    'success': '#2d6a4f',
    'warn':    '#e9c46a',
    'danger':  '#e76f51',
    'bg':      '#ffffff',
    'grid':    '#f1f3f5',
}

def _apply_style():
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         10,
        'axes.titlesize':    12,
        'axes.titleweight':  'bold',
        'axes.titlepad':     12,
        'axes.labelsize':    9,
        'axes.labelcolor':   PALETTE['mid'],
        'axes.edgecolor':    PALETTE['light'],
        'axes.facecolor':    PALETTE['bg'],
        'figure.facecolor':  PALETTE['bg'],
        'xtick.color':       PALETTE['mid'],
        'ytick.color':       PALETTE['mid'],
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'grid.color':        PALETTE['grid'],
        'grid.linewidth':    1,
    })


# ── Data preparation ──────────────────────────────────────────────────────────

def sessions_to_df(sessions):
    if not sessions:
        return pd.DataFrame()

    df = pd.DataFrame([dict(s) for s in sessions])
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_name'] = df['date'].dt.strftime('%a')  # Mon, Tue …
    df['hour'] = df['start_time'].apply(lambda t: int(t.split(':')[0]))

    def to_period(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['period'] = df['hour'].apply(to_period)
    return df


# ── Stats & insights ──────────────────────────────────────────────────────────

def compute_stats(df):
    if df.empty:
        return {}

    total_hours = round(df['duration'].sum(), 1)
    top_subject = df.groupby('subject')['duration'].sum().idxmax()
    avg_focus   = round(df['focus'].mean(), 1)
    best_period = df.groupby('period')['focus'].mean().idxmax()

    return {
        'total_hours': total_hours,
        'top_subject': top_subject,
        'avg_focus':   avg_focus,
        'best_period': best_period,
    }


def generate_insights(df):
    if df.empty or len(df) < 3:
        return ["Log more sessions to unlock personalized insights."]

    insights = []

    period_focus = df.groupby('period')['focus'].mean()
    best  = period_focus.idxmax()
    worst = period_focus.idxmin()
    insights.append(
        f"You focus best during {best.lower()} sessions "
        f"(avg {period_focus[best]:.1f}/5) and worst during {worst.lower()} "
        f"(avg {period_focus[worst]:.1f}/5)."
    )

    subj_focus = df.groupby('subject')['focus'].mean()
    if len(subj_focus) > 1:
        weakest  = subj_focus.idxmin()
        strongest = subj_focus.idxmax()
        insights.append(
            f"{strongest} is your strongest subject (focus {subj_focus[strongest]:.1f}/5). "
            f"{weakest} needs the most attention ({subj_focus[weakest]:.1f}/5) — "
            f"try scheduling it during {best.lower()}."
        )

    corr = df['mood'].corr(df['focus'])
    if abs(corr) > 0.4:
        direction = "positively" if corr > 0 else "negatively"
        insights.append(
            f"Mood and focus are {direction} correlated (r={corr:.2f}). "
            + ("When you feel good, you study well." if corr > 0
               else "You can sometimes push through even on low-mood days.")
        )

    avg_dur = df['duration'].mean()
    long_focus  = df[df['duration'] >  avg_dur]['focus'].mean() if (df['duration'] > avg_dur).any() else None
    short_focus = df[df['duration'] <= avg_dur]['focus'].mean() if (df['duration'] <= avg_dur).any() else None
    if long_focus and short_focus:
        if long_focus > short_focus + 0.3:
            insights.append(f"Longer sessions (>{avg_dur:.1f}h) tend to yield better focus for you.")
        elif short_focus > long_focus + 0.3:
            insights.append(f"Shorter, concentrated sessions work better for you than marathon ones.")

    return insights


def get_schedule_recommendations(df):
    if df.empty:
        return []

    rows = []
    for subject, group in df.groupby('subject'):
        period_focus = group.groupby('period')['focus'].mean()
        best = period_focus.idxmax()
        rows.append({
            'subject':    subject,
            'best_period': best,
            'avg_focus':  round(period_focus[best], 1),
        })

    rows.sort(key=lambda x: x['avg_focus'], reverse=True)
    return rows


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _save_fig(filename):
    path = os.path.join(CHARTS_DIR, filename)
    plt.tight_layout(pad=1.5)
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()


def _clean_axes(ax, show_grid_y=True):
    """Remove top/right spines, add subtle y-grid, clean up ticks."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(PALETTE['light'])
    ax.spines['bottom'].set_color(PALETTE['light'])
    if show_grid_y:
        ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color=PALETTE['grid'], zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ── Chart 1: Study hours by subject (horizontal bar) ─────────────────────────

def chart_hours_by_subject(df):
    filename = 'hours_by_subject.png'
    if df.empty:
        return filename

    _apply_style()
    data = df.groupby('subject')['duration'].sum().sort_values()

    fig, ax = plt.subplots(figsize=(6, max(2.8, len(data) * 0.55)))

    colors = [PALETTE['dark'] if v == data.max() else PALETTE['mid'] for v in data.values]
    bars = ax.barh(data.index, data.values, color=colors, edgecolor='none',
                   height=0.55, zorder=3)

    # value labels at end of each bar
    for bar, val in zip(bars, data.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}h', va='center', ha='left',
                fontsize=9, color=PALETTE['mid'])

    ax.set_xlabel('Total Hours')
    ax.set_title('Study Hours by Subject')
    ax.set_xlim(0, data.max() * 1.2)
    _clean_axes(ax, show_grid_y=False)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.8, color=PALETTE['grid'], zorder=0)
    ax.set_axisbelow(True)

    _save_fig(filename)
    return filename


# ── Chart 2: Avg focus by subject (colour-coded bars) ────────────────────────

def chart_avg_focus_by_subject(df):
    filename = 'focus_by_subject.png'
    if df.empty:
        return filename

    _apply_style()
    data = df.groupby('subject')['focus'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3.6))

    colors = [
        PALETTE['success'] if v >= 4
        else PALETTE['warn']   if v >= 3
        else PALETTE['danger']
        for v in data.values
    ]
    bars = ax.bar(data.index, data.values, color=colors, edgecolor='none',
                  width=0.55, zorder=3)

    ax.axhline(4, color=PALETTE['light'], linestyle='--', linewidth=1.2,
               label='Productive (≥4)', zorder=2)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel('Avg Focus (1–5)')
    ax.set_title('Average Focus by Subject')

    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                color=PALETTE['dark'])

    ax.legend(fontsize=8, framealpha=0)
    _clean_axes(ax)

    _save_fig(filename)
    return filename


# ── Chart 3: Weekly trend (line + shaded area) ───────────────────────────────

def chart_weekly_trend(df):
    filename = 'weekly_trend.png'
    if df.empty:
        return filename

    _apply_style()

    # group by ISO week, keep date label for x-axis
    weekly = (
        df.groupby(df['date'].dt.to_period('W'))
        .agg(total_hours=('duration', 'sum'))
        .reset_index()
    )
    weekly['label'] = weekly['date'].apply(lambda p: p.start_time.strftime('%b %d'))
    x = range(len(weekly))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(x, weekly['total_hours'], marker='o', color=PALETTE['dark'],
            linewidth=2, markersize=6, zorder=4)
    ax.fill_between(x, weekly['total_hours'], alpha=0.1, color=PALETTE['dark'])

    # annotate each point
    for i, val in enumerate(weekly['total_hours']):
        ax.annotate(f'{val:.1f}h',
                    xy=(i, val), xytext=(0, 8),
                    textcoords='offset points',
                    ha='center', fontsize=8, color=PALETTE['mid'])

    ax.set_xticks(list(x))
    ax.set_xticklabels(weekly['label'], rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Hours Studied')
    ax.set_title('Weekly Study Trend')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _clean_axes(ax)

    _save_fig(filename)
    return filename


# ── Chart 4: Focus by period (period emoji + bar) ────────────────────────────

def chart_focus_by_period(df):
    filename = 'focus_by_period.png'
    if df.empty:
        return filename

    _apply_style()

    order  = ['Morning', 'Afternoon', 'Evening', 'Night']

    data   = df.groupby('period')['focus'].mean().reindex(order).dropna()
    labels = list(data.index)

    fig, ax = plt.subplots(figsize=(6, 3.6))

    colors = [
        PALETTE['success'] if v >= 4
        else PALETTE['warn']   if v >= 3
        else PALETTE['danger']
        for v in data.values
    ]
    bars = ax.bar(labels, data.values, color=colors, edgecolor='none',
                  width=0.5, zorder=3)

    ax.axhline(4, color=PALETTE['light'], linestyle='--', linewidth=1.2,
               label='Productive (≥4)', zorder=2)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel('Avg Focus (1–5)')
    ax.set_title('Focus Level by Time of Day')

    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                color=PALETTE['dark'])

    ax.legend(fontsize=8, framealpha=0)
    _clean_axes(ax)

    _save_fig(filename)
    return filename


# ── Master function ───────────────────────────────────────────────────────────

def run_all(sessions):
    df = sessions_to_df(sessions)
    if df.empty:
        return {}, []

    stats = compute_stats(df)
    stats['insights'] = generate_insights(df)
    stats['schedule'] = get_schedule_recommendations(df)

    charts = [
        {'title': 'Study Hours by Subject',   'filename': chart_hours_by_subject(df)},
        {'title': 'Average Focus by Subject', 'filename': chart_avg_focus_by_subject(df)},
        {'title': 'Weekly Study Trend',       'filename': chart_weekly_trend(df)},
        {'title': 'Focus by Time of Day',     'filename': chart_focus_by_period(df)},
    ]

    return stats, charts
