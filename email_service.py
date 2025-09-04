import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os  # Added for environment variables
from dotenv import load_dotenv  # Added for environment variables

# Load environment variables (added)
load_dotenv()

def send_result_email(recipient_email, username, test_data):
    """Existing function - unchanged"""
    sender_email = "d60726851@gmail.com"
    sender_password = "pgsz zcck icuc qknx"  # Use Gmail App Password

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = "Your Fuel Quality Test Results"

    body = f"""
    <h2>Hello {username}!</h2>
    <p>Here are your fuel quality test results:</p>

    <table border="1">
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Fuel Type</td><td>{test_data['fuel_type']}</td></tr>
        <tr><td>Result</td><td>{test_data['result']}</td></tr>
        <tr><td>Confidence</td><td>{test_data['confidence']}</td></tr>
        <tr><td>Water Content</td><td>{test_data['water_content']} ppm</td></tr>
    </table>

    <p>Thank you for using our service!</p>
    """

    message.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

# NEW FUNCTION ADDED FOR PASSWORD RESET
def send_password_reset_email(recipient_email, reset_url):
    """New function for password reset emails"""
    sender_email = os.getenv('SMTP_USERNAME', "d60726851@gmail.com")  # Fallback to your email
    sender_password = os.getenv('SMTP_PASSWORD', "pgsz zcck icuc qknx")  # Fallback to your password

    message = MIMEMultipart()
    message["From"] = f"Fuel Analyzer <{sender_email}>"
    message["To"] = recipient_email
    message["Subject"] = "Password Reset Request"

    body = f"""
    <h2>Password Reset</h2>
    <p>You requested to reset your password. Click the link below:</p>
    <p><a href="{reset_url}">{reset_url}</a></p>
    <p><em>This link will expire in 1 hour.</em></p>
    <p>If you didn't request this, please ignore this email.</p>
    """

    message.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        print(f"Password reset email failed: {e}")
        return False
