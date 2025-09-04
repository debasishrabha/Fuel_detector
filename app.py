import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import joblib
import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from flask_migrate import Migrate
import secrets
import string
from itsdangerous import URLSafeTimedSerializer
# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_analyzer.db'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
db = SQLAlchemy(app)
migrate = Migrate(app, db)
# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    fuel_type = db.Column(db.String(50))
    viscosity = db.Column(db.Float)
    density = db.Column(db.Float)
    flash_point = db.Column(db.Float)
    sulphur_content = db.Column(db.Float)
    octane_content = db.Column(db.Float)
    water_content = db.Column(db.Float)
    result = db.Column(db.String(20))
    confidence = db.Column(db.String(20))
    days_until_degradation = db.Column(db.Integer)
    storage_grade = db.Column(db.String(20))
    degradation_factors = db.Column(db.String)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def send_test_results(recipient_email, username, test_data, result, degradation_info, issues=None, recommendations=None, warnings=None):
    import traceback
    from email.mime.base import MIMEBase
    from email import encoders
    import tempfile

    issues = issues or []
    recommendations = recommendations or []
    warnings = warnings or []

    sender_email = os.getenv('SMTP_USERNAME')
    sender_password = os.getenv('SMTP_PASSWORD')
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))

    if not all([sender_email, sender_password]):
        print("[ERROR] Email credentials are missing in .env")
        return False

    try:
        print(f"[INFO] Sending email from {sender_email} to {recipient_email}...")

        # Create email message
        message = MIMEMultipart()
        message["From"] = f"Fuel Analyzer <{sender_email}>"
        message["To"] = recipient_email
        message["Subject"] = f"Fuel Test Results: {result['prediction']}"

        # Create plain text version for attachment
        txt_report = f"""Fuel Quality Report for {username}

Test Result: {result['prediction']} ({result['confidence']})

--- Parameters ---
Fuel Type: {test_data['fuel_type']}
Viscosity: {test_data['viscosity']} cSt
Density: {test_data['density']} kg/m³
Flash Point: {test_data['flash_point']} °C
Sulphur: {test_data['sulphur_content']} ppm
Octane: {test_data['octane_content']} RON
Water: {test_data['water_content']} ppm

--- Storage Analysis ---
Days Until Degradation: {degradation_info[0]}
Storage Grade: {degradation_info[2]}
Degradation Factors: {', '.join(degradation_info[1]) if degradation_info[1] else 'None'}

--- Detected Issues ---
{chr(10).join(f"- {issue}" for issue in issues) if issues else "None"}

--- Recommendations ---
{chr(10).join(f"- {rec}" for rec in recommendations) if recommendations else "None"}

--- Warnings ---
{chr(10).join(f"- {warn}" for warn in warnings) if warnings else "None"}

Thank you for using Fuel Analyzer.
"""

        # HTML version (in email body)
        html_body = f"""
        <html>
        <body>
            <h2>Fuel Quality Analysis Report</h2>
            <p>Hello {username},</p>
            <p>Here are your test results:</p>

            <h3>Result: <span style="color:{'green' if result['prediction'] == 'PASS' else 'red'}">
            {result['prediction']} ({result['confidence']})
            </span></h3>

            <h4>Test Parameters</h4>
            <ul>
                <li>Fuel Type: {test_data['fuel_type']}</li>
                <li>Viscosity: {test_data['viscosity']} cSt</li>
                <li>Density: {test_data['density']} kg/m³</li>
                <li>Flash Point: {test_data['flash_point']} °C</li>
                <li>Sulphur: {test_data['sulphur_content']} ppm</li>
                <li>Octane: {test_data['octane_content']} RON</li>
                <li>Water: {test_data['water_content']} ppm</li>
            </ul>

            <h4>Storage Analysis</h4>
            <ul>
                <li>Days Until Degradation: {degradation_info[0]}</li>
                <li>Storage Grade: {degradation_info[2]}</li>
                <li>Degradation Factors: {', '.join(degradation_info[1]) if degradation_info[1] else 'None'}</li>
            </ul>

            <h4>Detected Issues</h4>
            <ul>
                {''.join(f"<li>{issue}</li>" for issue in issues) if issues else "<li>None</li>"}
            </ul>

            <h4>Recommendations</h4>
            <ul>
                {''.join(f"<li>{rec}</li>" for rec in recommendations) if recommendations else "<li>None</li>"}
            </ul>

            <h4>Warnings</h4>
            <ul>
                {''.join(f"<li>{warn}</li>" for warn in warnings) if warnings else "<li>None</li>"}
            </ul>

            <p>Thanks for using Fuel Analyzer!</p>
        </body>
        </html>
        """

        # Attach HTML body
        message.attach(MIMEText(html_body, "html"))

        # Creates a temporary text file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            f.write(txt_report)
            f.flush()
            file_path = f.name

        # Attach the .txt file
        with open(file_path, "rb") as file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename=Fuel_Report_{username}.txt")
            message.attach(part)

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()

            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())

        print("[SUCCESS] Email sent with attachment.")
        return True

    except Exception:
        print("[ERROR] Email sending failed:")
        traceback.print_exc()
    return False

class FuelQualityAnalyzer:
    def __init__(self):
        model_path = 'models/fuel_quality_model.pkl'
        scaler_path = 'models/fuel_quality_scaler.pkl'

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            print("Model not found. Training new model...")
            df = self.generate_sample_data()
            os.makedirs('models', exist_ok=True)
            self.model, self.scaler = self.train_fuel_quality_model(df)
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

    def generate_sample_data(self, num_samples=1000):
        np.random.seed(42)
        data = {
            'density': np.random.normal(820, 20, num_samples),
            'viscosity': np.random.normal(2.5, 0.5, num_samples),
            'sulphur_content': np.random.lognormal(1.2, 0.3, num_samples),
            'flash_point': np.random.normal(65, 10, num_samples),
            'octane_content': np.random.normal(92, 5, num_samples),
            'water_content': np.random.exponential(20, num_samples),
        }
        df = pd.DataFrame(data)
        conditions = (
                (df['density'].between(770, 850)) &
                (df['viscosity'].between(1.5, 4.0)) &
                (df['sulphur_content'] < 50) &
                (df['flash_point'] > 55) &
                (df['octane_content'] > 87) &
                (df['water_content'] < 100)
        )
        df['quality_pass'] = np.where(conditions, 1, 0)
        return df

    def train_fuel_quality_model(self, df):
        X = df.drop('quality_pass', axis=1)
        y = df['quality_pass']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        return model, scaler

    def predict_quality(self, input_data):
        input_df = pd.DataFrame([input_data])
        scaled_input = self.scaler.transform(input_df)
        prediction = self.model.predict(scaled_input)
        proba = self.model.predict_proba(scaled_input)

        return {
            'prediction': 'PASS' if prediction[0] == 1 else 'FAIL',
            'confidence': f"{max(proba[0]) * 100:.1f}%",
            'parameters': input_data
        }

def predict_fuel_degradation(fuel_type, parameters, storage_conditions):
    BASE_STABILITY = {
        'Petrol': 90,  # 3 months
        'Diesel': 180,  # 6 months
        'Kerosene': 270  # 9 months
    }

    degradation_factors = []
    degradation_rate = 1.0
    storage_score = 100

    # Water content impact
    water_impact = math.exp(parameters['water_content'] / 50 - 1)
    if parameters['water_content'] > 30:
        degradation_factors.append(f"High water content ({parameters['water_content']}ppm)")
    degradation_rate *= water_impact
    storage_score -= min(30, parameters['water_content'] * 0.5)

    # Sulphur content impact
    sulphur_impact = 1 + (parameters['sulphur_content'] / 100)
    if parameters['sulphur_content'] > 15:
        degradation_factors.append(f"Elevated sulphur ({parameters['sulphur_content']}ppm)")
    degradation_rate *= sulphur_impact

    # Viscosity impact
    if fuel_type == 'Diesel':
        viscosity_impact = 1 + abs(parameters['viscosity'] - 3.0) * 0.1
    else:
        viscosity_impact = 1 + abs(parameters['viscosity'] - 0.7) * 0.2
    degradation_rate *= viscosity_impact

    # Flash point impact
    if parameters['flash_point'] < (55 if fuel_type == 'Diesel' else -20):
        flash_impact = 1.5
        degradation_factors.append(f"Low flash point ({parameters['flash_point']}°C)")
    else:
        flash_impact = 1.0
    degradation_rate *= flash_impact

    # Octane/cetane impact
    octane_impact = 1.0
    if fuel_type == 'Petrol':
        octane_impact = 1 + (95 - parameters['octane_content']) * 0.02
        if parameters['octane_content'] < 95:
            degradation_factors.append(f"Lower octane ({parameters['octane_content']}RON)")
    elif fuel_type == 'Diesel':
        octane_impact = 1 + (50 - parameters.get('cetane_content', 50)) * 0.01
    degradation_rate *= octane_impact

    # Storage conditions impact
    temp = storage_conditions.get('temperature', 25)
    if temp > 30:
        temp_impact = math.exp((temp - 30) * 0.03)
        degradation_factors.append(f"High storage temp ({temp}°C)")
        storage_score -= (temp - 30) * 1.5
    elif temp < 10:
        temp_impact = 1 + (10 - temp) * 0.02
        degradation_factors.append(f"Low storage temp ({temp}°C)")
    else:
        temp_impact = 1.0
    degradation_rate *= temp_impact

    container_impact = {
        'metal': 1.0,
        'plastic': 1.3,
        'fiberglass': 1.2,
        'underground': 0.9
    }.get(storage_conditions.get('container_type', 'metal'), 1.0)
    degradation_rate *= container_impact

    headspace = storage_conditions.get('headspace', 10)
    headspace_impact = 1.0
    if headspace > 20:
        headspace_impact = 1 + (headspace - 20) * 0.01
        degradation_factors.append(f"Large air space ({headspace}%)")
    degradation_rate *= headspace_impact

    # Calculate final values
    base_stability = BASE_STABILITY.get(fuel_type, 90)
    days_until_degradation = base_stability / degradation_rate

    storage_grade = (
        "A (Excellent)" if storage_score >= 90 else
        "B (Good)" if storage_score >= 75 else
        "C (Fair)" if storage_score >= 60 else
        "D (Poor)" if storage_score >= 40 else
        "F (Unacceptable)"
    )

    return int(days_until_degradation), degradation_factors, storage_grade

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_email'] = user.email
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials!', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user_id' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))

    analyzer = FuelQualityAnalyzer()
    fuel_type = request.form.get('fuel_type') or 'Not Specified'  # Default value added

    input_data = {
        'density': float(request.form['density']),
        'viscosity': float(request.form['viscosity']),
        'sulphur_content': float(request.form['sulphur_content']),
        'flash_point': float(request.form['flash_point']),
        'octane_content': float(request.form['octane_content']),
        'water_content': float(request.form['water_content'])
    }

    result = analyzer.predict_quality(input_data)

    # Additional quality assessment
    quality_status = "Standard"
    issues = []
    recommendations = []
    warnings = []

    # Octane number check
    if input_data['octane_content'] < 87:
        issues.append(f"Low octane rating ({input_data['octane_content']} RON)")
        recommendations.append("Add MTBE-based octane booster to improve octane number")

    # Water content check
    if input_data['water_content'] > 50:
        issues.append(f"Elevated water content ({input_data['water_content']} ppm)")
        recommendations.append("Use a water separator or coalescing filter to remove water contamination")

    # Sulphur content check
    if input_data['sulphur_content'] > 50:
        issues.append(f"High sulphur content ({input_data['sulphur_content']} ppm)")
        recommendations.append("Consider desulphurization treatment or use sulphur-removing additives")
        warnings.append("High sulphur can damage emission control systems and increase pollution")

    # Flash point check
    if input_data['flash_point'] < 55:
        issues.append(f"Low flash point ({input_data['flash_point']} °C) - safety concern")
        recommendations.append("Check for contamination with lighter fractions. Consider blending with higher flash point fuel")
        warnings.append("Low flash point increases fire hazard during storage and handling")
    elif input_data['flash_point'] > 100:
        issues.append(f"High flash point ({input_data['flash_point']} °C) - may affect combustion")
        recommendations.append("Verify fuel composition. May need blending with lower flash point components")

    # Viscosity check
    if input_data['viscosity'] < 1.5:
        issues.append(f"Low viscosity ({input_data['viscosity']} cSt) - may cause pump wear")
        recommendations.append("Add viscosity improver or blend with higher viscosity fuel")
    elif input_data['viscosity'] > 4.0:
        issues.append(f"High viscosity ({input_data['viscosity']} cSt) - may affect atomization")
        recommendations.append("Consider preheating or blending with lower viscosity fuel")

    # Density check
    if not 770 <= input_data['density'] <= 850:
        issues.append(f"Density out of range ({input_data['density']} kg/m³)")
        recommendations.append("Verify fuel composition and consider blending adjustment")

    storage_conditions = {
        'temperature': float(request.form.get('storage_temp', 25)),
        'container_type': request.form.get('container_type', 'metal'),
        'headspace': float(request.form.get('headspace', 10))
    }

    degradation_info = predict_fuel_degradation(fuel_type, input_data, storage_conditions)
    if not degradation_info or len(degradation_info) < 3:
        degradation_info = (None, [], "Not Available")

    if issues:
        quality_status = "Adulterated"

    # Save to database
    new_result = TestResult(
        user_id=session['user_id'],
        fuel_type=fuel_type,
        viscosity=input_data['viscosity'],
        density=input_data['density'],
        flash_point=input_data['flash_point'],
        sulphur_content=input_data['sulphur_content'],
        octane_content=input_data['octane_content'],
        water_content=input_data['water_content'],
        result=result['prediction'],
        confidence=result['confidence'],
        days_until_degradation=degradation_info[0],
        storage_grade=degradation_info[2],
        degradation_factors=", ".join(degradation_info[1]) if degradation_info[1] else "None"
    )
    db.session.add(new_result)
    db.session.commit()

    # Send email to logged-in user
    email_sent = send_test_results(
        recipient_email=session['user_email'],
        username=session['username'],
        test_data={'fuel_type': fuel_type, **input_data},
        result=result,
        degradation_info=degradation_info,
        issues=issues,
        recommendations=recommendations,
        warnings=warnings
    )

    if not email_sent:
        flash('Failed to send email with results', 'warning')

    return render_template('result.html',
                       result=result,
                       fuel_type=fuel_type,
                       quality_status=quality_status,
                       issues=issues,
                       recommendations=recommendations,
                       warnings=warnings,
                       parameters=input_data,
                       days_until_degradation=degradation_info[0],
                       degradation_factors=degradation_info[1],
                       storage_grade=degradation_info[2])

@app.route('/test_email')
def test_email():
    if send_test_results(
        recipient_email="your@test.email",
        username="Test User",
        test_data={
            'fuel_type': 'Petrol',
            'viscosity': 2.5,
            'density': 820,
            'flash_point': 65,
            'sulphur_content': 15,
            'octane_content': 95,
            'water_content': 20
        },
        result={'prediction': 'PASS', 'confidence': '95.0%'},
        degradation_info=(90, ["High storage temp (35°C)"], "A (Excellent)")
    ):
        return "Email sent successfully!"
    return "Failed to send email"

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Initialize password reset serializer
def generate_token(email):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

def verify_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = serializer.loads(
            token,
            salt='password-reset-salt',
            max_age=expiration
        )
    except:
        return False
    return email

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate reset token
            token = generate_token(user.email)
            reset_url = url_for('reset_password', token=token, _external=True)
            
            # Send reset email
            send_password_reset_email(user.email, reset_url)
            
            flash('Password reset link has been sent to your email', 'info')
            return redirect(url_for('login'))
        
        flash('Email not found', 'danger')
    
    return render_template('forgot_password.html')

def send_password_reset_email(recipient_email, reset_url):
    sender_email = os.getenv('SMTP_USERNAME')
    sender_password = os.getenv('SMTP_PASSWORD')
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))

    if not all([sender_email, sender_password]):
        print("[ERROR] Email credentials are missing in .env")
        return False

    try:
        message = MIMEMultipart()
        message["From"] = f"Fuel Analyzer <{sender_email}>"
        message["To"] = recipient_email
        message["Subject"] = "Password Reset Request"

        html = f"""
        <html>
        <body>
            <h2>Password Reset Request</h2>
            <p>You requested to reset your password. Click the link below to reset it:</p>
            <p><a href="{reset_url}">{reset_url}</a></p>
            <p>This link will expire in 1 hour.</p>
            <p>If you didn't request this, please ignore this email.</p>
        </body>
        </html>
        """

        message.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())

        return True
    except Exception as e:
        print(f"Error sending password reset email: {e}")
        return False

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = verify_token(token)
    if not email:
        flash('Invalid or expired token', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(request.url)
        
        user = User.query.filter_by(email=email).first()
        if user:
            user.set_password(password)
            db.session.commit()
            flash('Password updated successfully!', 'success')
            return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
