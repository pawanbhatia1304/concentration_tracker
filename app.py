
import streamlit as st
import pandas as pd
import os
import glob
import io

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode
)

from streamlit_autorefresh import st_autorefresh

from core.camera import ConcentrationProcessor


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Concentration Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# SESSION STATE
# ==========================================

if "last_session_csv" not in st.session_state:
    st.session_state.last_session_csv = ""


# ==========================================
# CSS
# ==========================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0E1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #F8FAFC;
    }

    div[data-testid="stMetric"] {
        background-color: #1E293B;
        padding: 16px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# HELPER: GET SAVED SESSIONS
# ==========================================

def get_saved_sessions():

    os.makedirs("data/sessions", exist_ok=True)

    files = glob.glob(
        os.path.join("data", "sessions", "*.csv")
    )

    return sorted(
        files,
        key=os.path.getmtime,
        reverse=True
    )


# ==========================================
# HEADER
# ==========================================

st.title("🧠 Concentration Tracker")

st.caption(
    "AI-powered visual attention monitoring "
    "using Computer Vision and Deep Learning."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🧠 Concentration Tracker")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Session Analytics",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Deep Learning Mini Project"
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("Live Monitoring")

    st.info(
        "Press START to activate your webcam. "
        "Allow camera access in your browser."
    )

    ctx = webrtc_streamer(

        key="concentration-tracker",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=ConcentrationProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True

    )

    if ctx.state.playing:

        st_autorefresh(
            interval=1000,
            key="dashboard_refresh"
        )

    processor = ctx.video_processor

    metrics = None

    if processor is not None:

        metrics = processor.get_metrics()

        current_csv = processor.get_csv()

        if current_csv:
            st.session_state.last_session_csv = current_csv

    st.divider()

    st.subheader("Live Performance")

    if metrics:

        score = metrics["score"]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Visual Attention Score",
                f"{score:.1f}%"
                if score is not None
                else "—"
            )

        with col2:

            st.metric(
                "Processing FPS",
                metrics["fps"]
            )

        with col3:

            st.metric(
                "Current Status",
                metrics["status"]
            )

        if metrics["calibrated"]:

            st.success(
                "Calibration completed successfully."
            )

        else:

            st.warning(
                "Calibration in progress. "
                "Keep your eyes open and look at the screen."
            )

        st.divider()

        st.subheader("Behavioural Measurements")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("EAR", metrics["ear"])

        c2.metric("Gaze X", metrics["gaze"])

        c3.metric(
            "Yaw",
            f'{metrics["yaw"]}°'
        )

        c4.metric(
            "Pitch",
            f'{metrics["pitch"]}°'
        )

        c5, c6 = st.columns(2)

        c5.metric(
            "Blink Count",
            metrics["blinks"]
        )

        duration = metrics["duration"]

        c6.metric(
            "Session Duration",
            f"{int(duration // 60):02d}:"
            f"{int(duration % 60):02d}"
        )

    else:

        st.warning(
            "Start the webcam to display live measurements."
        )

    st.divider()

    st.subheader("Download Session")

    if st.session_state.last_session_csv:

        st.download_button(
            "📥 Download Current Session",
            data=st.session_state.last_session_csv,
            file_name="concentration_session.csv",
            mime="text/csv"
        )

    else:

        st.caption(
            "Start monitoring to generate session data."
        )

    st.caption(
        "This score estimates visual attentiveness. "
        "It does not directly measure mental concentration."
    )


# ==========================================
# SESSION ANALYTICS
# ==========================================

elif page == "Session Analytics":

    st.header("📊 Session Analytics")

    session_files = get_saved_sessions()

    if session_files:

        selected_file = st.selectbox(
            "Select a monitoring session",
            session_files,
            format_func=lambda path: os.path.basename(path)
        )

        try:

            df = pd.read_csv(selected_file)

            if not df.empty:

                st.subheader("Session Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Average Attention Score",
                    f'{df["visual_attention_score"].mean():.1f}%'
                )

                col2.metric(
                    "Total Recorded Samples",
                    len(df)
                )

                col3.metric(
                    "Total Blinks",
                    int(df["blink_count"].max())
                )

                st.divider()

                st.subheader("Visual Attention Trend")

                st.line_chart(
                    df.set_index("elapsed_seconds")[
                        ["visual_attention_score"]
                    ]
                )

                st.subheader("Eye Aspect Ratio")

                st.line_chart(
                    df.set_index("elapsed_seconds")[["ear"]]
                )

                st.subheader("Head Movement")

                st.line_chart(
                    df.set_index("elapsed_seconds")[
                        [
                            "yaw_relative",
                            "pitch_relative",
                            "roll_relative"
                        ]
                    ]
                )

                st.subheader("Recorded Data")

                st.dataframe(
                    df.tail(30),
                    use_container_width=True
                )

                st.download_button(
                    "📥 Download Selected Session",
                    data=df.to_csv(index=False),
                    file_name=os.path.basename(selected_file),
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Unable to read session: {error}"
            )

    else:

        st.info(
            "No saved sessions found. "
            "Complete a webcam monitoring session first."
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header("About Concentration Tracker")

    st.write(
        """
        Concentration Tracker is an AI-based application
        designed to estimate visual attentiveness through
        facial and eye-related behavioural analysis.

        Computer Vision is used to extract facial features,
        while a Deep Learning model will analyse sequential
        behavioural patterns.
        """
    )

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Streamlit",
        "OpenCV",
        "MediaPipe Face Mesh",
        "NumPy",
        "Pandas",
        "WebRTC",
        "TensorFlow / Keras"
    ]

    for technology in technologies:

        st.write(f"• {technology}")

    st.subheader("Main Features")

    st.write(
        """
        - Real-time face detection
        - Eye Aspect Ratio calculation
        - Gaze estimation
        - Head-pose estimation
        - Blink detection
        - Visual attention scoring
        - Session recording
        - Session analytics
        - LSTM-based sequential prediction (next stage)
        """
    )

    st.warning(
        "This is an educational prototype, "
        "not a medical or psychological diagnostic tool."
    )
