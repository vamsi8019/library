# RFID AI Smart Library Management System

A Streamlit-based RFID smart library application with AI-powered analytics, live scan-driven operations, and dashboard visualizations.

## Features
- RFID-only user and book operations
- Checkout, return, reservation, and registration by RFID scan
- AI analytics dashboard
- Demand prediction
- Availability prediction using Linear Regression, Holt-Winters, and Decision Tree Regressor
- Lost book detection
- Due-date violation prediction
- Real-time simulation mode
- Optional serial RFID reader support

## Run Locally
1. Install dependencies:
   ```bash
   c:/Users/USER/Downloads/LIB/.venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   c:/Users/USER/Downloads/LIB/.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503
   ```
3. Open:
   ```
   http://127.0.0.1:8503
   ```

## Main Files
- `app.py` - main RFID-first web application
- `streamlit_app.py` - earlier dashboard app
- `rfid_library_management.py` - core data generation and ML pipeline
- `requirements.txt` - Python dependencies

## Notes
- Use RFID tags in the form `USER-XXXXXXXXXX` and `BOOK-XXXXXXXXXX` for manual scan input.
- Serial reader input is supported through the sidebar RFID reader section.
