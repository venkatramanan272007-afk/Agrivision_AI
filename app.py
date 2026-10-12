
import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- GREEN THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f3f8f0;
}
[data-testid="stSidebar"] {
    background-color: #e1efdc;
}
h1, h2, h3 {
    color: #205a38;
}
.hero {
    background: linear-gradient(120deg, #174d32, #42935b);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    color: white !important;
}
.hero p {
    color: #eff9ed;
    font-size: 17px;
}
.card {
    background: white;
    border: 1px solid #dce9d7;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 10px;
}
div.stButton > button {
    background-color: #287747;
    color: white;
    border-radius: 10px;
    border: none;
    min-height: 44px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #174d32;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD TOMATO MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "tomato_disease_model.keras"
    )


@st.cache_data
def load_classes():
    return Path("class_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()


# ---------------- PLANT LIST ----------------
plants = [
    "Tomato",
    "Potato",
    "Chilli",
    "Brinjal",
    "Capsicum",
    "Mango",
    "Grapes",
    "Apple",
    "Cotton",
    "Rice",
    "Maize",
    "Banana",
    "Lemon",
    "Other"
]


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🌱 AgriVision AI")
    st.caption("SMART AGRICULTURE")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "About Project"]
    )

    st.divider()
    st.write("🌿 Plant Health")
    st.write("📷 Leaf Image Analysis")
    st.caption("Educational project | Version 1.0")


# ---------------- ABOUT PAGE ----------------
if page == "About Project":

    st.title("🌿 About AgriVision AI")

    st.write("""
    AgriVision AI is an agricultural website project
    designed to help users explore plant health using
    leaf images and an image-classification model.
    """)

    st.subheader("Main Features")

    st.markdown("""
    - Plant and crop selection
    - Leaf image upload
    - Tomato leaf class prediction
    - AI confidence display
    - Green agricultural dashboard
    """)

    st.info("""
    The current trained model supports only its
    existing tomato-related training classes.
    Other crops are listed for future development.
    """)


# ---------------- DASHBOARD ----------------
else:

    st.markdown("""
    <div class="hero">
        <h1>🌿 AgriVision Dashboard</h1>
        <p>
        Healthy Plants • Better Farming • Greener Future
        </p>
        <p>
        Select a crop and upload a leaf image to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
        <h3>🌱 Plants</h3>
        <h2>14 Categories</h2>
        <p>Crop selection</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <h3>🔍 AI Analysis</h3>
        <h2>Image Scan</h2>
        <p>Tomato model available</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <h3>📷 Image Upload</h3>
        <h2>JPG / PNG</h2>
        <p>Upload a clear leaf photo</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Crop selector
    st.subheader("🌾 Select a Plant / Crop")

    crop = st.selectbox(
        "Choose your plant",
        plants
    )

    if crop == "Tomato":
        st.success("Tomato AI model is available.")
    else:
        st.warning(
            f"{crop} is selected. Its disease detection "
            "model is not trained yet. Choose Tomato to "
            "use the current AI prediction feature."
        )

    st.divider()

    # Upload and results
    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    with left:
        st.subheader("📤 Upload Leaf Image")

        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"]
        )

        image = None

        if uploaded_file is not None:
            try:
                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    caption=f"Uploaded {crop} leaf",
                    use_container_width=True
                )

            except Exception:
                st.error("Unable to open this image.")

        detect = st.button(
            "🔍 Detect Leaf Condition",
            use_container_width=True
        )

    with right:
        st.subheader("📊 Prediction Result")

        if detect:

            if image is None:
                st.error(
                    "Please upload a leaf image first."
                )

            elif crop != "Tomato":
                st.error(
                    "The current AI model supports Tomato "
                    "only. Please select Tomato for prediction."
                )

            else:
                try:
                    with st.spinner(
                        "Analyzing tomato leaf..."
                    ):
                        model = load_model()
                        class_names = load_classes()

                        resized = image.resize((128, 128))

                        array = np.array(
                            resized,
                            dtype=np.float32
                        )

                        array = np.expand_dims(
                            array,
                            axis=0
                        )

                        predictions = model.predict(
                            array,
                            verbose=0
                        )[0]

                        if len(predictions) != len(class_names):
                            raise ValueError(
                                "Model output does not match "
                                "the saved class names."
                            )

                        index = int(
                            np.argmax(predictions)
                        )

                        disease = class_names[index]

                        confidence = float(
                            predictions[index] * 100
                        )

                    st.session_state["last_result"] = {
                        "crop": crop,
                        "disease": disease,
                        "confidence": confidence
                    }

                except Exception as error:
                    st.error(
                        f"Prediction failed: {error}"
                    )

        result = st.session_state.get("last_result")

        if result and result["crop"] == crop:

            disease = result["disease"]
            confidence = result["confidence"]

            st.success("Prediction completed!")

            st.markdown(
                f"### 🌿 {disease}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                max(0, min(100, int(confidence)))
            )

            if "healthy" in disease.lower():
                st.success(
                    "The model predicts the healthy class."
                )
            else:
                st.warning(
                    "The model predicts a possible disease "
                    "class. Please confirm with an agricultural "
                    "expert before taking action."
                )

            st.caption(
                "Confidence is not the same as real-world "
                "accuracy. Predictions may be incorrect."
            )

        else:
            st.info(
                "Upload a Tomato leaf image and click "
                "Detect Leaf Condition to view the result."
            )

    # ---------------- DISEASE EXAMPLES ----------------
    st.divider()
    st.subheader("🌿 Common Plant Health Problems")

    st.write(
        "Examples of plant diseases to support future "
        "multi-crop development:"
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""
        <div class="card">
        <h3>🍅 Tomato</h3>
        Early blight and late blight
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("""
        <div class="card">
        <h3>🥔 Potato</h3>
        Early blight and late blight
        </div>
        """, unsafe_allow_html=True)

    with d3:
        st.markdown("""
        <div class="card">
        <h3>🥭 Mango</h3>
        Anthracnose and other leaf problems
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "AgriVision AI | Educational prototype. "
        "Verify predictions with a qualified agricultural expert."
    )