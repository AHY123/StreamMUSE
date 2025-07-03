import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# --- Configuration ---
SENDER_EMAIL = "2564784746@qq.com"  # Your QQ email address
SENDER_PASSWORD = "Tp6m06vup@" # !!! Use your QQ Mail Authorization Code here !!!
RECEIVER_EMAIL = "2564784746@qq.com" # The Outlook recipient
SUBJECT = "MIDI File from QQ Mail"
BODY = "Here's the MIDI file sent from my QQ email."
MIDI_FILE_PATH = "/path/to/your/midi_file.midi" # Path to your MIDI file

# --- SMTP Server Details for QQ Mail ---
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465 # Port 465 is commonly used with SSL/TLS for QQ Mail

# --- Create the email message ---
msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL
msg['Subject'] = SUBJECT

# Attach the body
msg.attach(MIMEText(BODY, 'plain'))

# Attach the MIDI file
try:
    with open(MIDI_FILE_PATH, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',
                    f"attachment; filename= {os.path.basename(MIDI_FILE_PATH)}")
    msg.attach(part)
except FileNotFoundError:
    print(f"Error: MIDI file not found at {MIDI_FILE_PATH}")
    exit(1)
except Exception as e:
    print(f"Error attaching file: {e}")
    exit(1)

# --- Send the email ---
try:
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server: # Use SMTP_SSL for port 465
        # For port 587 with STARTTLS, you'd use:
        # server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        # server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
    print(f"Email with {os.path.basename(MIDI_FILE_PATH)} sent successfully to {RECEIVER_EMAIL} via QQ Mail!")
except smtplib.SMTPAuthenticationError:
    print("SMTP Authentication Error: Check your QQ email and Authorization Code.")
    print("Ensure you've enabled SMTP service in QQ Mail settings and generated an authorization code.")
except smtplib.SMTPConnectError as e:
    print(f"SMTP Connection Error: Could not connect to {SMTP_SERVER}:{SMTP_PORT}.")
    print(f"Error details: {e}")
    print("Check server address, port, and network connectivity.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")