# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 21:16:45 2025

@author: lalit
"""

import numpy as np
import pickle
import streamlit as st
from reportlab.pdfgen import canvas
import io
import plotly.graph_objects as go

# Load the saved model
Loaded_model = pickle.load(open('L:/logistic ml/logistic_model.sav', 'rb'))


# Sidebar Info
st.sidebar.title("❤️ Heart Health Info")
st.sidebar.markdown("""
**Heart disease** includes a range of conditions that affect the heart.

🔍 **Common types:**
- Coronary artery disease (CAD)
- Arrhythmia (abnormal heartbeat)
- Heart failure
- Congenital heart defects

⚠️ **Major risk factors:**
- High blood pressure
- High cholesterol
- Diabetes
- Smoking
- Obesity
- Physical inactivity

💡 **Prevention tips:**
- Eat a heart-healthy diet
- Exercise regularly
- Avoid tobacco and limit alcohol
- Manage stress effectively
- Get regular checkups
""")


# 📄 PDF Generation Function
def generate_pdf_report(prediction_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "🩺 Heart Disease Prediction Report")

    # Body
    c.setFont("Helvetica", 12)
    text_object = c.beginText(50, 770)
    for line in prediction_text.split('\n'):
        text_object.textLine(line.strip())
    c.drawText(text_object)

    c.save()
    buffer.seek(0)
    return buffer


# 📊 Graph Function
def show_risk_graph(age, trestbps, chol, fbs, thalach, oldpeak):
    labels = ['Age', 'Resting BP', 'Cholesterol', 'Fasting Sugar', 'Max HR', 'ST Depression']
    values = [age, trestbps, chol, fbs * 200, thalach, oldpeak * 50]
    thresholds = [50, 130, 240, 120, 120, 2]  # cutoff values

    colors = ['red' if val > th else 'green' for val, th in zip(values, thresholds)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{val}" for val in values],
        textposition='auto'
    ))

    fig.update_layout(
        title="Heart Risk Indicators",
        yaxis_title="Value (scaled where needed)",
        xaxis_title="Medical Parameters",
        height=400
    )

    st.plotly_chart(fig)


# Prediction Function
def heart_prediction(input_data, age, sex, cp, trestbps, chol, fbs, restecg,
                     thalach, exang, oldpeak, slope, ca, thal):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = Loaded_model.predict(input_data_reshaped)
    proba = Loaded_model.predict_proba(input_data_reshaped)[0][1] * 100

    if prediction[0] == 1:
        report = f"""
🩺 Prediction: High Risk of Heart Disease Detected

📝 Test Report:
Probability of Heart Disease: {proba:.2f}%

Key Input Values:
- Age: {age} {'(⚠️ High)' if age > 50 else ''}
- Sex: {'Male' if sex == 1 else 'Female'} {'(⚠️ Higher Risk)' if sex == 1 else ''}
- Chest Pain Type (cp): {cp} {'(⚠️ Atypical/Asymptomatic)' if cp >= 2 else ''}
- Resting BP (trestbps): {trestbps} mm Hg {'(⚠️ High)' if trestbps > 130 else ''}
- Cholesterol (chol): {chol} mg/dL {'(⚠️ High)' if chol > 240 else ''}
- Fasting Blood Sugar >120 (fbs): {'Yes' if fbs == 1 else 'No'} {'(⚠️ High)' if fbs == 1 else ''}
- Resting ECG (restecg): {restecg} {'(⚠️ Abnormal)' if restecg > 0 else ''}
- Max Heart Rate (thalach): {thalach} {'(⚠️ Low)' if thalach < 120 else ''}
- Exercise Induced Angina (exang): {'Yes' if exang == 1 else 'No'} {'(⚠️)' if exang == 1 else ''}
- ST Depression (oldpeak): {oldpeak} {'(⚠️ High)' if oldpeak > 2 else ''}
- Slope of ST (slope): {slope} {'(⚠️ Flat/Down)' if slope > 0 else ''}
- Major Vessels (ca): {ca} {'(⚠️ Blocked)' if ca > 0 else ''}
- Thalassemia (thal): {thal} {'(⚠️ Defect)' if thal > 1 else ''}

💡 Recommendations:
- Consult a Cardiologist for ECG, Echo, or Stress Test.
- Adopt a heart-healthy diet: low salt, low fat.
- Exercise moderately 5 days a week.
- Avoid smoking, alcohol, and stress.
- Monitor BP, cholesterol, sugar regularly.

⚠️ Disclaimer: This result is based on a machine learning model and is not a medical diagnosis. Always consult a licensed doctor.
"""
        st.error("🩺 **Prediction: High Risk of Heart Disease Detected**")
        return report
    else:
        report = f"""
✅ Prediction: No Heart Disease Detected

📝 Test Summary:
Probability of Heart Disease: {100 - proba:.2f}%

Keep maintaining your healthy lifestyle:
- Balanced diet
- Regular exercise
- Stress management
- Routine checkups

ℹ️ This tool provides a preliminary analysis based on inputs.
"""
        st.success("✅ **Prediction: No Heart Disease Detected**")
        return report


# Main function
def main():
    st.title("❤️ Heart Disease Prediction Web App")

    # Inputs
    age = st.number_input("Enter Your Age (years):", min_value=1, max_value=120, step=1)
    sex = st.number_input("Enter Your Sex (1=Male, 0=Female):", min_value=0, max_value=1, step=1)
    cp = st.number_input("Chest Pain Type (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic):", min_value=0, max_value=3, step=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg):", min_value=50, max_value=250, step=1)
    chol = st.number_input("Serum Cholesterol (mg/dL):", min_value=100, max_value=600, step=1)
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dL? (1=Yes, 0=No):", min_value=0, max_value=1, step=1)
    restecg = st.number_input("Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy):", min_value=0, max_value=2, step=1)
    thalach = st.number_input("Maximum Heart Rate Achieved:", min_value=60, max_value=250, step=1)
    exang = st.number_input("Exercise Induced Angina (1=Yes, 0=No):", min_value=0, max_value=1, step=1)
    oldpeak = st.number_input("ST Depression (oldpeak value):", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.number_input("Slope of ST segment (0=up, 1=flat, 2=down):", min_value=0, max_value=2, step=1)
    ca = st.number_input("Number of Major Vessels (0-3):", min_value=0, max_value=3, step=1)
    thal = st.number_input("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect):", min_value=1, max_value=3, step=1)

    diagnose = ''

    if st.button('Generate Heart Test Report'):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        diagnose = heart_prediction(input_data, age, sex, cp, trestbps, chol,
                                    fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

        # Show bar graph
        show_risk_graph(age, trestbps, chol, fbs, thalach, oldpeak)

    if diagnose:
        st.markdown("### 📋 Report Output:")
        st.markdown(diagnose)

        # PDF Download
        pdf_buffer = generate_pdf_report(diagnose)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name="Heart_Test_Report.pdf",
            mime="application/pdf"
        )
        


# Run the App
if __name__ == '__main__':
    main()
# Normal footer that appears at the bottom after scrolling
st.markdown(
    """
    <hr style="border:1px solid #ddd;">
    <div style="text-align: center; padding-top: 10px; font-size: 14px; color: gray;">
        Developed by <strong>Lalit Singh</strong> · Heart Disease Prediction · 2025<br>
        📧 Contact: <a href="mailto:youremail@example.com">lalitsinghs420@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)
