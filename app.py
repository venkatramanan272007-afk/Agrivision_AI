import hashlib
import hmac
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ----------------------------- Configuration -----------------------------
st.set_page_config(
    page_title="AgriVision | Plant Health",
    page_icon="🌿",
    layout="wide",
)

MODEL_PATH = Path("tomato_disease_model.keras")
CLASS_PATH = Path("class_names.txt")
DB_PATH = Path("agrivision_reports.db")

# Demo-only login. Do not use a real or reused password in this demo app.
DEMO_USERS = {
    "student": hashlib.sha256("agrivision123".encode("utf-8")).hexdigest()
}


# ----------------------------- Database ----------------------------------
def init_db():
    """Create the local SQLite table if it does not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_name TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                scan_date TEXT NOT NULL
            )
            """
        )


def save_report(plant_name: str, result: str, confidence: float):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO scan_reports (plant_name, result, confidence, scan_date)
            VALUES (?, ?, ?, ?)
            """,
            (
                plant_name,
                result,
                float(confidence),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )


def get_reports() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM scan_reports ORDER BY id DESC", conn
        )


init_db()


# ----------------------------- Model helpers -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.resolve()}. "
            "Place tomato_disease_model.keras in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_classes():
    if not CLASS_PATH.is_file():
        raise FileNotFoundError(
            f"Class file not found: {CLASS_PATH.resolve()}. "
            "Create class_names.txt with one class name per line, in the same "
            "order used when training the model."
        )
    classes = [
        line.strip()
        for line in CLASS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError("class_names.txt is empty.")
    return classes


def prep_image(image: Image.Image, model):
    """Resize and scale an image for a conventional Keras image classifier.

    IMPORTANT: This uses 0..1 pixel scaling. If the model was trained with a
    different preprocessing pipeline, reproduce that pipeline here.
    """
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) != 4:
        raise ValueError(
            f"Expected model input shape (batch, height, width, channels), got {shape}."
        )

    height = int(shape[1]) if shape[1] is not None else 224
    width = int(shape[2]) if shape[2] is not None else 224
    channels = int(shape[3]) if shape[3] is not None else 3

    if channels not in (1, 3):
        raise ValueError(
            f"The model expects {channels} channels. This app supports 1 or 3 channels."
        )

    image = image.convert("L" if channels == 1 else "RGB")
    image = image.resize((width, height))
    array = np.asarray(image, dtype=np.float32)

    if channels == 1:
        array = array[..., np.newaxis]

    # Change this if your training pipeline used different normalization.
    array = array / 255.0
    return np.expand_dims(array, axis=0), shape


def predict_leaf(image: Image.Image):
    model = load_model()
    classes = load_classes()
    array, input_shape = prep_image(image, model)
    scores = np.asarray(model.predict(array, verbose=0)[0]).reshape(-1)

    if len(scores) != len(classes):
        raise ValueError(
            f"The model returned {len(scores)} scores, but class_names.txt has "
            f"{len(classes)} class names. Check that the file matches the model."
        )

    # This assumes the model returns class probabilities (e.g. softmax output).
    # If it returns logits, apply the matching output conversion used in training.
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index]) * 100.0
    predicted_class = classes[best_index].replace("_", " ")
    return predicted_class, confidence, classes, scores, array.shape, input_shape


# ----------------------------- Plant library ------------------------------
PLANTS = [
    ("Vegetables", "Tomato", "Solanum lycopersicum", "🍅",
     "The current AI model is intended for the classes in the trained tomato dataset."),
    ("Vegetables", "Potato", "Solanum tuberosum", "🥔",
     "Library information only; no potato AI model is included."),
    ("Vegetables", "Chilli", "Capsicum annuum", "🌶️",
     "Library information only; no chilli AI model is included."),
    ("Vegetables", "Brinjal", "Solanum melongena", "🍆",
     "Library information only; no brinjal AI model is included."),
    ("Fruits", "Mango", "Mangifera indica", "🥭",
     "Evergreen fruit tree with long leaves."),
    ("Fruits", "Banana", "Musa", "🍌",
     "Large herbaceous plant with broad leaves."),
    ("Fruits", "Grapevine", "Vitis vinifera", "🍇",
     "Look at both sides of leaves for unusual spots."),
    ("Fruits", "Apple", "Malus domestica", "🍎",
     "Library information only; no apple AI model is included."),
    ("Trees", "Neem", "Azadirachta indica", "🌳",
     "Common South Asian tree with compound leaves."),
    ("Trees", "Mango tree", "Mangifera indica", "🌳",
     "Evergreen tree with long, narrow leaves."),
    ("Trees", "Coconut palm", "Cocos nucifera", "🌴",
     "Palm with long feather-like fronds."),
    ("Trees", "Banyan", "Ficus benghalensis", "🌳",
     "Large tree known for aerial roots."),
    ("Trees", "Teak", "Tectona grandis", "🌳",
     "Deciduous tree with large leaves."),
    ("Crops", "Rice", "Oryza sativa", "🌾",
     "Grass crop with long, narrow leaves."),
    ("Crops", "Maize", "Zea mays", "🌽",
     "Crop with broad blade-shaped leaves."),
    ("Crops", "Cotton", "Gossypium", "🌱",
     "Library information only; no cotton AI model is included."),
    ("Houseplants", "Tulsi", "Ocimum tenuiflorum", "🌿",
     "Aromatic herb with opposite leaves."),
    ("Houseplants", "Money plant", "Epipremnum aureum", "🪴",
     "Climbing houseplant with heart-shaped leaves."),
]


# ----------------------------- Styling ------------------------------------
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #edf8ee, #d9f0dc, #f7fbf5);
        color: #183b2a;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123d2b, #1b5e3a);
    }
    [data-testid="stSidebar"] * { color: white; }
    h1, h2, h3 { color: #175c36; }
    .hero {
        background: linear-gradient(120deg, #174b32, #398653);
        padding: 26px;
        border-radius: 20px;
        color: white;
        margin-bottom: 20px;
    }
    .hero h1, .hero p { color: white !important; }
    .hero p { font-size: 17px; }
    .info-card {
        background: white;
        border: 1px solid #c7e5cc;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .info-card h3 { color: #17613a; }
    .stButton > button {
        background: #207447;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        border: none;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #72ad80;
        border-radius: 14px;
        padding: 12px;
    }
    .footer {
        text-align: center;
        color: #55745e;
        padding: 20px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Login --------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

if not st.session_state["logged_in"]:
    st.markdown(
        '<div class="hero"><h1>🌿 Welcome to AgriVision</h1>'
        "<p>Browse plants and trees, view photos and videos, and try tomato "
        "leaf analysis.</p></div>",
        unsafe_allow_html=True,
    )

    login_tab, demo_tab = st.tabs(["🔐 Login", "Demo access"])

    with login_tab:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            normalized_username = username.strip()
            digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
            expected_digest = DEMO_USERS.get(normalized_username)

            if expected_digest and hmac.compare_digest(digest, expected_digest):
                st.session_state["logged_in"] = True
                st.session_state["username"] = normalized_username
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with demo_tab:
        st.info("Demo username: student | Password: agrivision123")
        st.warning("Demo login only. Do not use a personal password.")

    st.stop()


# ----------------------------- Navigation ---------------------------------
with st.sidebar:
    st.markdown("# 🌿 AgriVision")
    st.caption("SMART PLANT HEALTH PLATFORM")
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Plant & Tree Library",
            "Photo & Video Gallery",
            "AI Leaf Analysis",
            "Scan Reports",
            "About Project",
        ],
    )
    st.markdown("---")
    st.write(f"Signed in as **{st.session_state['username']}**")
    if st.button("Log out"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()


# ----------------------------- Pages --------------------------------------
if page == "Dashboard":
    st.markdown(
        '<div class="hero"><h1>🌿 AgriVision Plant Health</h1>'
        "<p>Plant library, photo/video gallery and tomato leaf AI demo.</p></div>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Plant library", f"{len(PLANTS)} entries")
    col2.metric("AI model", "Tomato only")
    col3.metric("Media", "Photos + videos")

    st.markdown("## Browse categories")
    columns = st.columns(4)
    for column, (emoji, name) in zip(
        columns,
        [("🌳", "Trees"), ("🍃", "Leaves & plants"),
         ("🥬", "Vegetables"), ("🍎", "Fruits")],
    ):
        with column:
            st.markdown(
                f'<div class="info-card"><h3>{emoji} {name}</h3>'
                "<p>Choose Plant & Tree Library from the sidebar to browse.</p></div>",
                unsafe_allow_html=True,
            )
    st.info(
        "The AI can only predict classes it was trained on. Adding plants to "
        "the library does not automatically teach the model to identify them."
    )


elif page == "Plant & Tree Library":
    st.markdown(
        '<div class="hero"><h1>🌳 Plant & Tree Library</h1>'
        "<p>Browse vegetables, fruit plants, trees, crops and houseplants.</p></div>",
        unsafe_allow_html=True,
    )
    categories = ["All"] + sorted({plant[0] for plant in PLANTS})
    category = st.selectbox("Browse category", categories)
    search = st.text_input(
        "Search by plant or scientific name",
        placeholder="Example: neem, mango, tomato",
    ).strip().casefold()

    found = [
        plant for plant in PLANTS
        if (category == "All" or plant[0] == category)
        and (search in plant[1].casefold() or search in plant[2].casefold())
    ]

    if not found:
        st.warning("No matching plant found.")
    else:
        for start in range(0, len(found), 3):
            columns = st.columns(3)
            for column, plant in zip(columns, found[start:start + 3]):
                with column:
                    st.markdown(
                        f'<div class="info-card"><h3>{plant[3]} {plant[1]}</h3>'
                        f"<p><i>{plant[2]}</i></p><p>{plant[4]}</p>"
                        f"<p><b>Category:</b> {plant[0]}</p></div>",
                        unsafe_allow_html=True,
                    )


elif page == "Photo & Video Gallery":
    st.markdown(
        '<div class="hero"><h1>📷 Photo & Video Gallery</h1>'
        "<p>Preview plant photos and videos uploaded from your device.</p></div>",
        unsafe_allow_html=True,
    )
    photos = st.file_uploader(
        "Choose plant photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="gallery_photos",
    )
    if photos:
        columns = st.columns(3)
        for index, uploaded_file in enumerate(photos):
            with columns[index % 3]:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                except (UnidentifiedImageError, OSError):
                    st.warning(f"Could not open {uploaded_file.name}.")

    videos = st.file_uploader(
        "Choose plant videos",
        type=["mp4", "mov", "webm"],
        accept_multiple_files=True,
        key="gallery_videos",
    )
    if videos:
        for video in videos:
            st.video(video)


elif page == "AI Leaf Analysis":
    st.markdown(
        '<div class="hero"><h1>🧠 AI Tomato Leaf Analysis</h1>'
        "<p>Upload a tomato leaf photo and run the trained classification model.</p></div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "This model is only suitable for the classes it was trained on. "
        "A prediction is not a confirmed plant-health diagnosis."
    )

    uploaded = st.file_uploader(
        "Upload a tomato leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        key="analysis_image",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Selected leaf photo", use_container_width=True)

            if st.button("🔍 Analyze Tomato Leaf", use_container_width=True):
                with st.spinner("Loading model and analyzing image..."):
                    predicted_class, confidence, classes, scores, array_shape, input_shape = (
                        predict_leaf(image)
                    )

                st.success(f"Predicted class: {predicted_class}")
                st.metric("Model score", f"{confidence:.2f}%")
                st.progress(float(np.clip(confidence / 100.0, 0.0, 1.0)))
                st.caption(
                    f"Processed image shape: {array_shape} | "
                    f"Model input shape: {input_shape}"
                )

                save_report("Tomato", predicted_class, confidence)

                with st.expander("Show all class scores"):
                    ranked_scores = sorted(
                        zip(classes, scores),
                        key=lambda item: float(item[1]),
                        reverse=True,
                    )
                    for class_name, score in ranked_scores:
                        st.write(
                            f"{class_name.replace('_', ' ')}: "
                            f"{float(score) * 100:.2f}%"
                        )
                st.caption(
                    "The displayed score reflects the model output; it is not "
                    "the same as real-world accuracy."
                )
        except Exception as error:
            st.error(
                "Could not analyze this image. Check that the model file, "
                "class names, and preprocessing match the training setup."
            )
            st.code(str(error))
    else:
        st.info("Upload a JPG, JPEG, PNG or WEBP image to begin.")


elif page == "Scan Reports":
    st.markdown(
        '<div class="hero"><h1>📊 Plant Scan Reports</h1>'
        "<p>Review previous tomato leaf predictions saved by this app.</p></div>",
        unsafe_allow_html=True,
    )
    reports = get_reports()
    st.metric("Total scans", len(reports))

    if not reports.empty:
        st.dataframe(reports, use_container_width=True, hide_index=True)
        csv_data = reports.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name="agrivision_scan_reports.csv",
            mime="text/csv",
        )
    else:
        st.info("No scan reports available yet.")


elif page == "About Project":
    st.markdown(
        '<div class="hero"><h1>🌿 About AgriVision</h1>'
        "<p>A student mini project for plant education and tomato leaf classification.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("## Aim")
    st.write(
        "Provide a simple website to browse plants and trees, view user-uploaded "
        "media, and test a trained tomato leaf classifier."
    )
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        '<div class="info-card"><h3>🐍 Python</h3><p>Application logic</p></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        '<div class="info-card"><h3>🧠 TensorFlow</h3><p>Runs the trained classifier</p></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        '<div class="info-card"><h3>🌐 Streamlit</h3><p>Website interface</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Limitations and future work")
    st.write("- Login is a demo only and is not production authentication.")
    st.write("- AI predictions are limited to the classes used to train the tomato model.")
    st.write("- Supporting more crops requires labelled datasets, training and evaluation.")
    st.write("- Photos and videos are uploaded by the user; the app does not fetch media automatically.")


st.markdown(
    '<div class="footer">🌿 AgriVision | BCA Mini Project | Plant Library & Tomato Leaf Analysis</div>',
    unsafe_allow_html=True,
)
