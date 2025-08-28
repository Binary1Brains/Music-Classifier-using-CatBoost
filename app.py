import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
import re

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon=":musical_note:",
    layout="wide"
)

if 'users' not in st.session_state:
    st.session_state['users'] = {
        "admin@example.com": {
            "name": "Admin User",
            "password": "password123"
        }
    }

def login_page():
    st.header("Login to Your Account")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            users = st.session_state.get('users', {})
            if email in users and users[email]['password'] == password:
                st.session_state['logged_in'] = True
                st.session_state['user_info'] = users[email]
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Incorrect email or password.")

def signup_page():
    st.header("Create a New Account")
    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Sign Up")

        if submitted:
            users = st.session_state.get('users', {})
            if not (name and email and password and confirm_password):
                st.warning("Please fill out all fields.")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.warning("Please enter a valid email address.")
            elif email in users:
                st.error("An account with this email already exists.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users[email] = {"name": name, "password": password}
                st.session_state['users'] = users
                st.success("Account created successfully! Please go to the Login page to log in.")

def main_app():
    st.sidebar.header(f"Welcome, {st.session_state['user_info']['name']}!")
    if st.sidebar.button("Logout"):
        del st.session_state['logged_in']
        del st.session_state['user_info']
        st.rerun()

    st.title("Music Genre Classification App")
    st.write("Upload an audio file (MP3/WAV) and let the model predict its genre.")

    @st.cache_data
    def load_assets():
        with open("catboost_genre_classifier.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f: label_encoder = pickle.load(f)
        return model, scaler, label_encoder

    try:
        model, scaler, label_encoder = load_assets()
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'catboost_genre_classifier.pkl', 'scaler.pkl', and 'label_encoder.pkl' are in the same directory.")
        return

    def extract_features(y, sr):
        features = {}
        features['length'] = len(y)/sr
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma)
        features['chroma_stft_var'] = np.var(chroma)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spec_cent)
        features['spectral_centroid_var'] = np.var(spec_cent)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spec_bw)
        features['spectral_bandwidth_var'] = np.var(spec_bw)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)
        y_harmonic = librosa.effects.harmonic(y)
        features['harmony_mean'] = np.mean(y_harmonic)
        features['harmony_var'] = np.var(y_harmonic)
        X = librosa.stft(y)
        frequencies = librosa.fft_frequencies(sr=sr)
        perceptual = librosa.perceptual_weighting(np.abs(X)**2, frequencies)
        features['perceptr_mean'] = np.mean(perceptual)
        features['perceptr_var'] = np.var(perceptual)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features['tempo'] = tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_var'] = np.var(mfcc[i])
        return pd.DataFrame([features])

    uploaded_file = st.file_uploader("Upload your music file", type=["mp3", "wav"])

    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=None)
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner("Analyzing and Visualizing..."):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Waveform")
                fig, ax = plt.subplots()
                librosa.display.waveshow(y, sr=sr, ax=ax, color="purple")
                st.pyplot(fig)

                st.subheader("MFCCs")
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                fig, ax = plt.subplots()
                img = librosa.display.specshow(mfccs, x_axis="time", sr=sr, ax=ax, cmap="coolwarm")
                fig.colorbar(img, ax=ax)
                st.pyplot(fig)

            with col2:
                st.subheader("Spectrogram")
                X_stft = librosa.stft(y)
                Xdb = librosa.amplitude_to_db(abs(X_stft))
                fig, ax = plt.subplots()
                img = librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)

            st.subheader("Prediction")
            with st.spinner("Extracting features and predicting genre..."):
                features_df = extract_features(y, sr)
                X_scaled = scaler.transform(features_df)
                y_pred = model.predict(X_scaled)
                predicted_genre = label_encoder.inverse_transform(y_pred)[0]
                st.success(f"The predicted genre is: **{predicted_genre}**")
    else:
        st.info("Upload a music file to get started")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main_app()
else:
    st.title("Welcome to the Music Genre Classifier")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose an action", ["Login", "Sign Up"])

    if page == "Login":
        login_page()
    elif page == "Sign Up":
        signup_page()

