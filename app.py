import streamlit as st
from src.predict import check_message
import time

st.set_page_config(
    page_title="Ayush(fake link detection) | UPI Fraud Shield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e9ecef !important;
        padding: 15px !important;
        font-size: 16px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #4A90E2 !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #4A90E2 0%, #003366 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        letter-spacing: 1px;
        border: none;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
    }

    .fraud-card {
        background-color: #fff3f3;
        border-left: 6px solid #ff4b4b;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.1);
        margin-top: 20px;
    }
    .safe-card {
        background-color: #f2fcf5;
        border-left: 6px solid #2e8b57;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #1e293b; margin-bottom: 0px;'>🛡️Ayush's Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 18px; margin-top: -10px;'>Advanced UPI & Link Fraud Detection</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("**Paste the suspicious SMS, WhatsApp message, or Email below:**")
user_input = st.text_area(
    label="Message Input",
    label_visibility="collapsed",
    height=160,
    placeholder="e.g., Dear user, your HDFC bank account is blocked. Update KYC immediately at http://bit.ly/update-kyc-now to avoid penalty."
)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 Analyze Threat Level", type="primary"):
    
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        with st.spinner("Engaging NLP Engine... Analyzing URL structures..."):
            time.sleep(1.5)
            result = check_message(user_input)
            
        st.markdown("---")
        
        if result == "FRAUD":
            st.markdown("""
                <div class="fraud-card">
                    <h2 style="color: #ff4b4b; margin-top: 0;">🚨 CRITICAL THREAT DETECTED</h2>
                    <p style="color: #333; font-size: 16px;">This message contains patterns heavily associated with <b>UPI Scams</b> and <b>Phishing</b>.</p>
                    <ul style="color: #555;">
                        <li><b>Do not</b> click on any embedded links.</li>
                        <li><b>Do not</b> download any APKs or files.</li>
                        <li><b>Do not</b> enter your UPI PIN to "receive" money.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        elif result == "SAFE":
            st.markdown("""
                <div class="safe-card">
                    <h2 style="color: #2e8b57; margin-top: 0;">✅ SAFE TO PROCEED</h2>
                    <p style="color: #333; font-size: 16px;">No immediate threats detected.</p>
                    <p style="color: #555; font-size: 14px;"><i>Note: While our NLP engine found no malicious keywords or suspicious URL structures, always remain vigilant when sharing personal information online.</i></p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        else:
            st.error(result)

st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("ℹ️ How does this system work?"):
    st.markdown("""
    **Implementation Details:**
    * **NLP Message Analyzer:** Extracts text features using TF-IDF vectorization, looking for urgency indicators and financial lures.
    * **URL Structure Analysis:** Scans for obfuscated links, suspicious shorteners (like bit.ly), and IP-based routing.
    * **Machine Learning Engine:** Powered by a Random Forest Classifier trained on thousands of known scam parameters.
    """)