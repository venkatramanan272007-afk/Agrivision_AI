
import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AgriVision | Plant Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PROFESSIONAL GREEN THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #edf8ee 0%, #d9f0dc 50%, #f4fbf3 100%);
    color: #183b2a;
}

header[data-testid="stHeader"] {
    background: rgba(237, 248, 238, 0.9);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123d2b 0%, #1b5e3a 100%);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.06);
    padding: 8px;
    border-radius: 8px;
}

h1, h2, h3 {
    color: #175c36 !important;
}

p, label, .stMarkdown {
    color: #254b35;
}

.hero {
    background: linear-gradient(120deg, #174b32, #28794b);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(20, 70, 40, 0.13);
}

.hero h1, .hero h2, .hero p {
    color: white !important;
}

.hero p {
    font-size: 17px;
}

.info-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #c7e5cc;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 4px 14px rgba(22, 78, 43, 0.06);
    margin-bottom: 12px;
}

.info-card h3 {
    margin-top: 0;
    color: #17613a !important;
}

.info-card p {
    margin-bottom: 0;
    color: #496653;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #c7e5cc;
    padding: 18px;
    border-radius: 14px;
}

.stButton > button {
    background: #207447;
    color: white;
    border: 1px solid #207447;
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #145531;
    color: white;
    border-color: #145531;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7);
    border: 1px dashed #72ad80;
    border-radius: 14px;
    padding: 14px;
}

div[data-baseweb="select"] > div {
    background: #ffffff;
    border-radius: 10px;
}

.result-card {
    background: #ffffff;
    border-left: 6px solid #25834b;
    border-radius: 12px;
    padding: 20px;
    margin-top: 12px;
}

.footer {
    text-align: center;
    color: #55745e;
    padding: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato_disease_model.keras")


@st.cache_data
def load_classes():
    return Path("class_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()


# ---------------- PLANT OPTIONS ----------------
plants = [
    "Tomato", "Potato", "Chilli", "Brinjal",
    "Capsicum", "Mango", "Grapes", "Apple",
    "Cotton", "Rice", "Maize", "Banana",
    "Lemon", "Other"
]


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🌿 AgriVision")
    st.caption("SMART PLANT HEALTH PLATFORM")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard", "About Project"]
    )

    st.markdown("---")
    st.markdown("### 🌱 Project Status")
    st.write("Tomato model: Available")
    st.write("Other crops: Coming soon")

    st.markdown("---")
    st.caption("Helping plants grow healthier")


# ---------------- ABOUT PAGE ----------------
if page == "About Project":

    st.markdown("""
    <div class="hero">
        <h1>🌿 About AgriVision</h1>
        <p>A plant health detection project designed to help
        users identify possible tomato leaf diseases.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🎯 Project Aim")
    st.write(
        "To provide a simple website where users can upload "
        "a tomato leaf image and receive a model prediction."
    )

    st.markdown("## ⚙️ Technologies Used")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
        <h3>🐍 Python</h3>
        <p>Programming language used for the project.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
        <h3>🧠 TensorFlow</h3>
        <p>Runs the trained image classification model.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
        <h3>🌐 Streamlit</h3>
        <p>Creates the interactive website interface.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Note: The current trained model supports the classes "
        "listed in class_names.txt. Other crops are not trained yet."
    )


# ---------------- DASHBOARD PAGE ----------------
else:

    st.markdown("""
    <div class="hero">
        <h1>🌿 AgriVision Plant Health Dashboard</h1>
        <p>Upload a tomato leaf image to check its predicted class.</p>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
        <h3>🍃 Plant Analysis</h3>
        <p>Upload a clear leaf image for a model prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
        <h3>🔎 Disease Classification</h3>
        <p>View the class predicted by the trained model.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
        <h3>🌎 Crop Support</h3>
        <p>More crop models can be added in future versions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🪴 Analyze Your Plant")

    left, right = st.columns([1, 1], gap="large")

    with left:
        crop = st.selectbox(
            "Choose your plant",
            plants
        )

        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"]
        )

        st.caption(
            "For best results, upload a clear image of one tomato leaf."
        )

    with right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded leaf image",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div class="info-card">
            <h3>📷 Image Preview</h3>
            <p>Your selected leaf image will appear here.</p>
            </div>
            """, unsafe_allow_html=True)

    if crop != "Tomato":
        st.warning(
            f"The current model has not been trained for {crop}. "
            "Please choose Tomato to use the available model."
        )

    elif uploaded_file is not None:
        if st.button("🔍 Analyze Tomato Leaf", use_container_width=True):

            try:
                with st.spinner("Analyzing image..."):
                    model = load_model()
                    class_names = load_classes()

                    # Standard image preprocessing
                    resized_image = image.resize((224, 224))
                    image_array = np.array(
                        resized_image,
                        dtype=np.float32
                    )

                    # Many image classifiers expect pixel values in [0, 1].
                    # Change this only if your model was trained differently.
                    image_array = image_array / 255.0
                    image_array = np.expand_dims(image_array, axis=0)

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    if len(predictions) != len(class_names):
                        st.error(
                            "The model output does not match class_names.txt. "
                            "Please check the class order and file."
                        )
                    else:
                        predicted_index = int(np.argmax(predictions))
                        predicted_class = class_names[predicted_index]
                        confidence = float(predictions[predicted_index]) * 100

                        st.markdown("""
                        <div class="result-card">
                        <h3>🌱 Prediction Result</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        st.success(
                            f"Predicted class: {predicted_class}"
                        )

                        st.metric(
                            "Model confidence",
                            f"{confidence:.2f}%"
                        )

                        st.progress(
                            min(max(confidence / 100, 0.0), 1.0)
                        )

                        st.caption(
                            "This is a model prediction, not a confirmed "
                            "diagnosis. Confidence is not the same as "
                            "real-world accuracy. Check with an agricultural "
                            "expert before treating a plant."
                        )

            except Exception as error:
                st.error(
                    "Could not analyze the image. Check that the model "
                    "and class_names.txt are available and compatible."
                )
                st.code(str(error))


    # Disease classes
    st.markdown("---")
    st.markdown("## 🌿 Current Model Classes")

    st.write(
        "The following are the classes available to your current model. "
        "Exact predictions depend on how the model was trained."
    )

    try:
        class_names = load_classes()
        cols = st.columns(3)

        for index, name in enumerate(class_names):
            with cols[index % 3]:
                st.markdown(
                    f"""
                    <div class="info-card">
                    <h3>🍃 {name.replace('_', ' ')}</h3>
                    <p>Class in the current trained model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception:
        st.info("Class list will appear after class_names.txt is available.")

    st.markdown("""
    <div class="footer">
        🌿 AgriVision | BCA Mini Project | Plant Health Detection
    </div>
    """, unsafe_allow_html=True)
