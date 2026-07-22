# AI Study Pattern Analyzer

A web application that helps university students track their study sessions and understand their study habits through data analysis and machine learning.

---

## Motivation

I built this during my second year at university after realizing I had no idea whether my study sessions were actually productive or just hours spent staring at a screen. I wanted something that could tell me *when* I study best, *which subjects* I struggle with, and ideally predict whether a planned session was worth starting at all.

Most productivity apps are either too generic or too complex. This one is built specifically around the way university students study — by subject, in blocks, with varying energy levels throughout the day.

---

## Features

- **Session logging** — record subject, date, start time, duration, mood (1–5), and focus level (1–5)
- **Dashboard analytics** — total hours, most-studied subject, average focus, best study period
- **Four charts** — hours by subject, average focus by subject, weekly trend, focus by time of day
- **Personalized insights** — natural-language observations generated from your data (e.g. "You focus best during evening sessions")
- **Schedule recommendations** — per-subject suggestions based on when you historically focus best
- **ML predictions** — a Random Forest classifier predicts whether a planned session will be productive, with a confidence percentage
- **Feature importance** — the dashboard shows which inputs (mood, time of day, subject, duration) drive the model's predictions

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| Database | SQLite |
| Data analysis | Pandas, NumPy |
| Machine learning | Scikit-learn (RandomForestClassifier) |
| Charts | Matplotlib |
| Frontend | Bootstrap 5, Jinja2 templates |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI-Study-Pattern-Analyzer.git
cd AI-Study-Pattern-Analyzer

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

**Optional:** seed the database with sample data to see the dashboard immediately:

```bash
python -c "from database import seed_sample_data; seed_sample_data()"
```

---

## Project Structure

```
AI-Study-Pattern-Analyzer/
├── app.py              # Flask routes
├── database.py         # SQLite setup and query functions
├── analytics.py        # Pandas analysis and Matplotlib charts
├── model.py            # Random Forest model — train, predict, feature importance
├── requirements.txt
├── study.db            # SQLite database (created on first run)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── log.html
│   ├── sessions.html
│   ├── dashboard.html
│   └── predict.html
└── static/
    ├── css/style.css
    └── charts/         # Matplotlib PNGs saved here
```

---

## Screenshots

| Page | Description |
|---|---|
| Home | Landing page with recent sessions |
| Log Session | Form to record a study session |
| My Sessions | Full history table with delete |
| Dashboard | Stats, charts, insights, ML model info, schedule |
| Predict | Form to predict productivity of a planned session |

*(Screenshots coming — run the app locally to see it)*

---

## Machine Learning Approach

The app uses a **Random Forest Classifier** from scikit-learn to predict whether a study session will be productive, defined as focus ≥ 4.

**Features used:**
- Hour of day (extracted from start time)
- Subject (label-encoded)
- Mood (1–5)
- Duration (hours)

**Training:**
- The model trains on all logged sessions using a 75/25 train/test split
- It retrains automatically each time the dashboard is visited, so it always reflects the latest data
- Accuracy and feature importances are displayed on the dashboard

**Why Random Forest?**
It handles small datasets reasonably well, requires no feature scaling, and produces interpretable feature importances — which is useful here since the goal is insight, not just prediction accuracy.

**Limitation:** with fewer than ~20 sessions the model won't be very accurate. The more you log, the better it gets.

---

## Limitations

- No user accounts — designed as a single-user local application
- The ML model needs a reasonable number of sessions (20+) before predictions become meaningful
- Charts are static PNGs regenerated on each dashboard visit rather than interactive
- No data export — sessions live only in the local SQLite database
- The productivity threshold (focus ≥ 4) is fixed and not configurable

---

## Future Improvements

These are things I'd realistically add with more time:

- **Pomodoro timer** — log sessions directly from a built-in 25-minute timer
- **PDF reports** — export a weekly summary as a downloadable PDF
- **Calendar view** — visualize study sessions on a monthly calendar grid
- **GPA prediction** — correlate study patterns with actual grades over time
- **Mobile app** — a simple React Native version for logging on the go
- **Interactive charts** — replace Matplotlib PNGs with Chart.js for hover tooltips
- **Configurable productivity threshold** — let users decide what "productive" means to them
- **Dark mode** — Bootstrap makes this straightforward with `data-bs-theme`

---

## Author

Built as a portfolio project — second year CS student exploring data analysis and applied ML.
