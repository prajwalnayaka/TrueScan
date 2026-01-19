from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from urllib.parse import urlencode
import click
from flask.cli import with_appcontext
from config import Config
from database import get_db_connection, create_tables, is_doctor_approved, add_approved_doctor, generate_doctor_id, get_scan_by_id
from test import run_all_models, calculate_overall_prediction
from auth0_service import create_auth0_oauth, requires_auth, get_user_info

app = Flask(__name__,template_folder=r'D:\TrueScan\templates',static_folder=r'D:\TrueScan\static')
app.config.from_object(Config)
auth0 = create_auth0_oauth(app)


@app.route('/')
def home():
    if session.get('profile'):
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/login')
def login():
    return auth0.authorize_redirect(redirect_uri=url_for('callback', _external=True))

@app.route('/admin')
@requires_auth
def admin():
    user = get_user_info()
    admin_emails = ['pragyamvikram@gmail.com', 'prajwalnayakat@gmail.com']

    if user['email'] not in admin_emails:
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('dashboard'))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT email, hospital_name FROM approved_doctors WHERE status = 'active' ORDER BY email")
    approved_doctors = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('admin.html', approved_doctors=approved_doctors)


@app.route('/admin/add', methods=['POST'])
@requires_auth
def add_doctor_admin():
    user = get_user_info()
    admin_emails = ['pragyamvikram@gmail.com', 'prajwalnayakat@gmail.com']

    if user['email'] not in admin_emails:
        return redirect(url_for('dashboard'))

    email = request.form['email']
    hospital = request.form.get('hospital', 'Hospital')

    if add_approved_doctor(email, hospital):
        flash(f'Successfully added {email}')
    else:
        flash(f'Failed to add {email}. Email might already exist.')

    return redirect(url_for('admin'))


@app.route('/callback')
def callback():
    token = auth0.authorize_access_token()
    user_info = token.get('userinfo')

    if not user_info:
        return redirect(url_for('home'))

    user_email = user_info.get('email', '')
    auth0_user_id = user_info['sub']

    if not is_doctor_approved(user_email):
        session.clear()
        flash('Access denied. Contact administrator for access.')
        return render_template('unauthorized.html')

    session['profile'] = {
        'user_id': auth0_user_id,
        'name': user_info.get('name', ''),
        'email': user_email,
        'picture': user_info.get('picture', '')
    }

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, doctor_id FROM users WHERE auth0_user_id = %s", (auth0_user_id,))
    user = cur.fetchone()

    if not user:
        doctor_id = generate_doctor_id()
        cur.execute("INSERT INTO users (doctor_id, auth0_user_id, name, email) VALUES (%s, %s, %s, %s)",
                    (doctor_id, auth0_user_id, user_info.get('name', ''), user_email))
        session['profile']['doctor_id'] = doctor_id
        flash(f'Welcome! Your Doctor ID: {doctor_id}')
    else:
        session['profile']['doctor_id'] = user[1]

    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/models')
@requires_auth
def models_page():
    # In a real application, this data could come from a database or a config file.
    models_info = [
        {
            'name': 'YOLOv8m-cls',
            'type': 'Object Detection & Classification',
            'description': 'A state-of-the-art, highly efficient, real-time object detection model, fine-tuned for classifying medical scan authenticity with high speed and accuracy.',
            'details':'Parameters ≈ 16.9M',
            'release':'Release Year: 2023'
        },
        {
            'name': 'ResNet50',
            'type': 'Residual Neural Network',
            'description': 'A deep convolutional neural network with 50 layers, renowned for its powerful feature extraction capabilities and robustness in image classification tasks.',
            'details': 'Parameters ≈ 23.5M',
            'release': 'Release Year: 2015'
        },
        {
            'name': 'VGG19_BN',
            'type': 'Convolutional Neural Network',
            'description': 'A 19-layer deep learning model known for its excellent performance in large-scale image recognition, providing a strong baseline for authenticity verification. The vanilla VGG19 was released in 2014.',
            'details':'Parameters ≈ 138M',
            'release': 'Release Year: 2020'
        }
    ]
    return render_template('models.html', models=models_info, user=get_user_info())


@app.route('/logout')
def logout():
    session.clear()
    params = {'returnTo': url_for('home', _external=True), 'client_id': Config.AUTH0_CLIENT_ID}
    return redirect(f"https://{Config.AUTH0_DOMAIN}/v2/logout?" + urlencode(params))


@app.route('/dashboard')
@requires_auth
def dashboard():
    return render_template('dashboard.html', user=get_user_info())


@app.route('/process_scan', methods=['POST'])
@requires_auth
def process_scan():
    if 'scan_file' not in request.files or request.files['scan_file'].filename == '':
        flash('No file selected. Please choose a scan to analyze.')
        return redirect(url_for('dashboard'))

    file = request.files['scan_file']
    user = get_user_info()

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename

    # --- MODIFIED CODE ---
    upload_folder = os.path.join(app.static_folder, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    model_results = run_all_models(filepath)
    overall_prediction = calculate_overall_prediction(model_results)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE auth0_user_id = %s", (user['user_id'],))
    user_id = cur.fetchone()[0]

    # --- IMPORTANT CHANGE FOR DATABASE ---
    # We need to store a web-accessible relative path in the database, not the full system path.
    db_filepath = os.path.join('static/uploads', filename).replace('\\', '/') # Use forward slashes for web paths

    cur.execute("""INSERT INTO scans (user_id, filename, file_path, gpu_predictions, 
                   colab_predictions, overall_fake_probability, processed)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (user_id, filename, db_filepath, json.dumps(model_results),
                 json.dumps([]), overall_prediction, True))

    scan_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for('analysis', scan_id=scan_id))


@app.route('/analysis/<int:scan_id>')
@requires_auth
def analysis(scan_id):
    user = get_user_info()
    scan_data = get_scan_by_id(scan_id, user['user_id'])

    if not scan_data:
        flash('Analysis not found or you do not have permission to view it.')
        return redirect(url_for('dashboard'))

    return render_template('analysis.html', scan=scan_data, user=user)


# --- END OF NEW ROUTE ---

@app.route('/report/<int:scan_id>')
@requires_auth
def report_form(scan_id):
    return render_template('report_form.html', scan_id=scan_id, user=get_user_info())


@app.route('/add_doctor/<email>')
def add_doctor_quick(email):
    if add_approved_doctor(email, "Hospital"):
        return f"✅ Added {email} - <a href='/'>Go to app</a>"
    else:
        return f"❌ Failed to add {email} - <a href='/'>Go to app</a>"


# --- Database Commands ---
@click.command(name='init-db')
@with_appcontext
def init_db_command():
    create_tables()
    click.echo('Initialized the database.')


app.cli.add_command(init_db_command)


@click.command(name='reset-db')
@with_appcontext
def reset_db_command():
    conn = get_db_connection()
    cur = conn.cursor()
    print("Dropping all tables...")
    cur.execute("DROP TABLE IF EXISTS reports, scans, users, approved_doctors CASCADE;")
    conn.commit()
    cur.close()
    conn.close()
    init_db_command()


app.cli.add_command(reset_db_command)

if __name__ == '__main__':
    app.run(debug=True)
