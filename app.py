from flask import Flask, render_template, request, redirect, url_for, flash
from database import init_db, add_session, get_all_sessions, get_session_count, delete_session
from analytics import run_all
from model import train, predict, get_feature_importance
from datetime import date

app = Flask(__name__)
app.secret_key = 'study-analyzer-secret'

with app.app_context():
    init_db()


@app.route('/')
def index():
    session_count = get_session_count()
    recent_sessions = get_all_sessions()[:5]
    return render_template('index.html', session_count=session_count, recent_sessions=recent_sessions)


@app.route('/log', methods=['GET', 'POST'])
def log_session():
    if request.method == 'POST':
        subject  = request.form.get('subject', '').strip()
        date_val = request.form.get('date', '')
        start    = request.form.get('start_time', '')
        duration = request.form.get('duration', '')
        mood     = request.form.get('mood', '')
        focus    = request.form.get('focus', '')

        if not all([subject, date_val, start, duration, mood, focus]):
            flash('Please fill in all fields.', 'danger')
            return redirect(url_for('log_session'))

        try:
            duration = float(duration)
            mood     = int(mood)
            focus    = int(focus)
        except ValueError:
            flash('Duration, mood, and focus must be numbers.', 'danger')
            return redirect(url_for('log_session'))

        if not (1 <= mood <= 5) or not (1 <= focus <= 5):
            flash('Mood and focus must be between 1 and 5.', 'danger')
            return redirect(url_for('log_session'))

        if duration <= 0:
            flash('Duration must be greater than 0.', 'danger')
            return redirect(url_for('log_session'))

        add_session(subject, date_val, start, duration, mood, focus)
        flash(f'Session logged! {duration}h of {subject} added.', 'success')
        return redirect(url_for('sessions'))

    today = date.today().isoformat()
    return render_template('log.html', today=today)


@app.route('/sessions')
def sessions():
    all_sessions = get_all_sessions()
    return render_template('sessions.html', sessions=all_sessions)


@app.route('/delete/<int:session_id>', methods=['POST'])
def delete(session_id):
    delete_session(session_id)
    flash('Session deleted.', 'info')
    return redirect(url_for('sessions'))


@app.route('/dashboard')
def dashboard():
    all_sessions = get_all_sessions()

    if not all_sessions:
        return render_template('dashboard.html', stats={}, charts=[], model_info={}, importances=[])

    stats, charts = run_all(all_sessions)

    # retrain the model every time the dashboard is loaded so it always reflects latest data
    model_info = train(all_sessions)
    importances = get_feature_importance(all_sessions)

    return render_template('dashboard.html',
                           stats=stats,
                           charts=charts,
                           model_info=model_info,
                           importances=importances)


@app.route('/predict', methods=['GET', 'POST'])
def predict_session():
    result = None
    subjects = []

    # pull the list of known subjects for the dropdown
    all_sessions = get_all_sessions()
    seen = set()
    for s in all_sessions:
        if s['subject'] not in seen:
            subjects.append(s['subject'])
            seen.add(s['subject'])

    if request.method == 'POST':
        subject  = request.form.get('subject', '').strip()
        start    = request.form.get('start_time', '')
        mood     = request.form.get('mood', '')
        duration = request.form.get('duration', '')

        if not all([subject, start, mood, duration]):
            flash('Please fill in all fields.', 'danger')
        else:
            try:
                result = predict(subject, start, int(mood), float(duration))
            except Exception as e:
                flash(f'Prediction error: {e}', 'danger')

    return render_template('predict.html', result=result, subjects=subjects)


if __name__ == '__main__':
    app.run(debug=True)
