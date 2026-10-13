import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import hashlib
import hmac

st.set_page_config(page_title="AgriVision | Plant Health", page_icon="🌿", layout="wide")

MODEL_PATH = Path("tomato_disease_model.keras")
CLASS_PATH = Path("class_names.txt")

# Demo-only login. Do not use real passwords here.
DEMO_USERS = {"student": hashlib.sha256("agrivision123".encode()).hexdigest()}

DB_PATH = "agrivision_reports.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_name TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL,
                scan_date TEXT NOT NULL
            )
        """)

def save_report(plant_name, result, confidence):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO scan_reports
               (plant_name, result, confidence, scan_date)
               VALUES (?, ?, ?, ?)""",
            (
                plant_name,
                result,
                float(confidence),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

def get_reports():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM scan_reports ORDER BY id DESC",
            conn
        )

init_db()
st.markdown("""
<style>
/* Main green background */
.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #edf8ee, #d9f0dc, #f7fbf5) !important;
    color: #183b2a !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123d2b, #1b5e3a) !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Headings */
h1, h2, h3 {
    color: #175c36 !important;
}

/* Welcome banner */
.hero {
    background: linear-gradient(120deg, #174b32, #398653);
    padding: 28px;
    border-radius: 20px;
    color: white;
    margin-bottom: 20px;
}
.hero h1, .hero p {
    color: white !important;
}
.hero p {
    font-size: 17px;
}

/* Login and Demo tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #dcecdf !important;
    padding: 6px;
    border-radius: 12px;
    gap: 8px;
}
.stTabs button[data-baseweb="tab"] {
    background: #ffffff !important;
    color: #174b32 !important;
    border-radius: 8px;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background: #287747 !important;
    color: #ffffff !important;
}
.stTabs button[data-baseweb="tab"] p {
    color: inherit !important;
}

/* Input boxes */
.stTextInput label {
    color: #174b32 !important;
}
.stTextInput input {
    background: #ffffff !important;
    color: #183b2a !important;
    border: 1px solid #72ad80 !important;
}

/* Buttons */
.stButton > button {
    background: #207447 !important;
    color: #ffffff !important;
    border-radius: 10px;
    font-weight: 600;
    border: none;
}

/* Upload box */
div[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px dashed #72ad80;
    border-radius: 14px;
    padding: 12px;
}

/* Information cards */
.info-card {
    background: #ffffff;
    border: 1px solid #c7e5cc;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
}
.info-card h3 {
    color: #17613a !important;
}

/* Messages */
.stAlert p {
    color: #183b2a !important;
}

.footer {
    text-align: center;
    color: #55745e;
    padding: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource

def reports_page():
    st.header("📊 Plant Scan Reports")

    reports = get_reports()
    st.metric("Total Scans", len(reports))

    if not reports.empty:
        st.dataframe(reports, use_container_width=True)

        csv_data = reports.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name="agrivision_scan_reports.csv",
            mime="text/csv"
        )
    else:
        st.info("No scan reports available yet.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Missing class file: {CLASS_PATH}")
    return [x.strip() for x in CLASS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


def prep_image(image, model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) != 4:
        raise ValueError(f"Expected image model input (batch,height,width,channels), got {shape}")
    h = shape[1] if isinstance(shape[1], int) else 224
    w = shape[2] if isinstance(shape[2], int) else 224
    channels = shape[3] if isinstance(shape[3], int) else 3
    if channels not in (1, 3):
        raise ValueError(f"Model expects {channels} channels (input shape {shape}); normal photos have 3 RGB channels. Check the saved model.")
    image = image.convert("RGB").resize((w, h))
    arr = np.asarray(image, dtype=np.float32)
    if channels == 1:
        arr = np.asarray(image.convert("L").resize((w, h)), dtype=np.float32)[..., np.newaxis]
    # This assumes training used pixel values scaled to 0..1.
    arr = arr / 255.0
    return np.expand_dims(arr, 0), shape


PLANTS = [
    ("Vegetables","Tomato","Solanum lycopersicum","🍅","The current AI model is for the classes in the trained tomato dataset."),
    ("Vegetables","Potato","Solanum tuberosum","🥔","Library information only; no potato AI model is included."),
    ("Vegetables","Chilli","Capsicum annuum","🌶️","Library information only; no chilli AI model is included."),
    ("Vegetables","Brinjal","Solanum melongena","🍆","Library information only; no brinjal AI model is included."),
    ("Fruits","Mango","Mangifera indica","🥭","Evergreen fruit tree with long leaves."),
    ("Fruits","Banana","Musa","🍌","Large herbaceous plant with broad leaves."),
    ("Fruits","Grapevine","Vitis vinifera","🍇","Look at both sides of leaves for unusual spots."),
    ("Fruits","Apple","Malus domestica","🍎","Library information only; no apple AI model is included."),
    ("Trees","Neem","Azadirachta indica","🌳","Common South Asian tree with compound leaves."),
    ("Trees","Mango tree","Mangifera indica","🌳","Evergreen tree with long, narrow leaves."),
    ("Trees","Coconut palm","Cocos nucifera","🌴","Palm with long feather-like fronds."),
    ("Trees","Banyan","Ficus benghalensis","🌳","Large tree known for aerial roots."),
    ("Trees","Teak","Tectona grandis","🌳","Deciduous tree with large leaves."),
    ("Crops","Rice","Oryza sativa","🌾","Grass crop with long, narrow leaves."),
    ("Crops","Maize","Zea mays","🌽","Crop with broad blade-shaped leaves."),
    ("Crops","Cotton","Gossypium","🌱","Library information only; no cotton AI model is included."),
    ("Houseplants","Tulsi","Ocimum tenuiflorum","🌿","Aromatic herb with opposite leaves."),
    ("Houseplants","Money plant","Epipremnum aureum","🪴","Climbing houseplant with heart-shaped leaves."),
]

# Login gate
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""
  
if not st.session_state.get("logged_in", False):
    st.markdown(
        '<div class="hero"><h1>🌿 Welcome to AgriVision</h1>'
        '<p>Browse plants and trees, open photos and videos, '
        'and try tomato leaf analysis.</p></div>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["🔐 Login", "Demo access"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            digest = hashlib.sha256(password.encode()).hexdigest()

            if username.strip() in DEMO_USERS and hmac.compare_digest(
                digest, DEMO_USERS[username.strip()]
            ):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with tab2:
        st.info("Demo username: student | Password: agrivision123")
        st.warning("Demo login only; do not use a personal password.")

    st.stop()
with st.sidebar:
    st.markdown("# 🌿 AgriVision")
    st.caption("SMART PLANT HEALTH PLATFORM")
    page = st.radio("Navigation", ["Dashboard","Plant & Tree Library","Photo & Video Gallery","AI Leaf Analysis","About Project"])
    st.markdown("---")
    st.write(f"Signed in as **{st.session_state.username}**")
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

if page == "Dashboard":
    st.markdown('<div class="hero"><h1>🌿 AgriVision Plant Health</h1><p>Plant library, photo/video gallery and tomato leaf AI demo.</p></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("Plant library", f"{len(PLANTS)} entries")
    b.metric("AI model", "Tomato only")
    c.metric("Media", "Photos + videos")
    st.markdown("## Browse categories")
    cols = st.columns(4)
    for col, (emoji, name) in zip(cols,[("🌳","Trees"),("🍃","Leaves & plants"),("🥬","Vegetables"),("🍎","Fruits")]):
        with col:
            st.markdown(f'<div class="info-card"><h3>{emoji} {name}</h3><p>Choose Plant & Tree Library from the sidebar to browse.</p></div>', unsafe_allow_html=True)
    st.info("The AI can only predict classes it was trained on. Adding plants to the library does not automatically teach the model to identify them.")

elif page == "Plant & Tree Library":
    st.markdown('<div class="hero"><h1>🌳 Plant & Tree Library</h1><p>Browse vegetables, fruit plants, trees, crops and houseplants.</p></div>', unsafe_allow_html=True)
    categories = ["All"] + sorted({p[0] for p in PLANTS})
    category = st.selectbox("Browse category", categories)
    search = st.text_input("Search by plant or scientific name", placeholder="Example: neem, mango, tomato")
    found = [p for p in PLANTS if (category == "All" or p[0] == category) and (search.lower() in p[1].lower() or search.lower() in p[2].lower())]
    if not found:
        st.warning("No matching plant found.")
    for start in range(0,len(found),3):
        cols = st.columns(3)
        for col,p in zip(cols,found[start:start+3]):
            with col:
                st.markdown(f'<div class="info-card"><h3>{p[3]} {p[1]}</h3><p><i>{p[2]}</i></p><p>{p[4]}</p><p><b>Category:</b> {p[0]}</p></div>', unsafe_allow_html=True)

elif page == "Photo & Video Gallery":
    st.markdown('<div class="hero"><h1>📷 Photo & Video Gallery</h1><p>Open plant photos and videos from your device.</p></div>', unsafe_allow_html=True)
    photos = st.file_uploader("Choose plant photos", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
    if photos:
        cols = st.columns(3)
        for i, f in enumerate(photos):
            with cols[i % 3]:
                try:
                    st.image(Image.open(f).convert("RGB"), caption=f.name, use_container_width=True)
                except Exception:
                    st.warning(f"Could not open {f.name}")
    st.divider()
    video = st.file_uploader("Choose a plant video", type=["mp4","mov","webm"])
    if video:
        st.video(video)
    st.caption("This gallery displays files you select. It does not automatically search the internet.")

elif page == "AI Leaf Analysis":
    st.markdown('<div class="hero"><h1>🔬 AI Leaf Analysis</h1><p>Upload a tomato leaf photo to run the available classifier.</p></div>', unsafe_allow_html=True)
    st.warning("The current model supports only the classes in your tomato training dataset. Other plants can be browsed but not diagnosed by this model.")
    uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg","jpeg","png"], key="ai_upload")
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Selected leaf photo", use_container_width=True)
            if st.button("🔍 Analyze Tomato Leaf", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    model = load_model()
                    classes = load_classes()
                    arr, input_shape = prep_image(image, model)
                    st.caption(f"Input image shape: {arr.shape} | Model expects: {input_shape}")
                    scores = np.asarray(model.predict(arr, verbose=0)[0]).reshape(-1)
                    if len(scores) != len(classes):
                        st.error(f"Model output has {len(scores)} values, but class_names.txt has {len(classes)} names. Check that both files match.")
                    else:
                        idx = int(np.argmax(scores))
                        confidence = float(scores[idx]) * 100
                        predicted_class = classes[idx].replace("_", " ")
                        st.success(f"Predicted class: {predicted_class}")
                        save_report(
                        plant_name="Tomato",
                        result=predicted_class,
                        confidence=confidence
                        )
                        st.metric("Model score", f"{confidence:.2f}%")
                        st.progress(float(np.clip(confidence/100,0,1)))
                        with st.expander("Show all class scores"):
                            for name, score in sorted(zip(classes,scores), key=lambda x: float(x[1]), reverse=True):
                                st.write(f"{name.replace('_',' ')}: {float(score)*100:.2f}%")
                        st.caption("This is a model prediction, not a confirmed diagnosis. Model score is not the same as real-world accuracy.")
        except Exception as e:
            st.error("Could not analyze this image. Check the model file and training preprocessing.")
            st.code(str(e))
    else:
            except Exception as e:
            st.error("Could not analyze this image. Check the model file and training preprocessing.")
            st.code(str(e))

    else:
        st.info("Upload a JPG, JPEG or PNG image to begin.")
       

elif page == "About Project":
    st.markdown('<div class="hero"><h1>🌿 About AgriVision</h1><p>A student mini project for plant education and tomato leaf classification.</p></div>', unsafe_allow_html=True)
    st.markdown("## Aim")
    st.write("Provide a simple website to browse plants and trees, view user-uploaded media, and test a trained tomato leaf classifier.")
    x,y,z = st.columns(3)
    x.markdown('<div class="info-card"><h3>🐍 Python</h3><p>Application logic</p></div>', unsafe_allow_html=True)
    y.markdown('<div class="info-card"><h3>🧠 TensorFlow</h3><p>Runs the trained classifier</p></div>', unsafe_allow_html=True)
    z.markdown('<div class="info-card"><h3>🌐 Streamlit</h3><p>Website interface</p></div>', unsafe_allow_html=True)
    st.markdown("## Limitations and future work")
    st.write("- Login is a demo only and has no database.")
    st.write("- AI predictions are limited to the classes used to train the current tomato model.")
    st.write("- Supporting more crops requires suitable labelled datasets, model training and evaluation.")
    st.write("- Photos/videos are uploaded by the user; the app does not automatically fetch media from the web.")

st.markdown('<div class="footer">🌿 AgriVision | BCA Mini Project | Plant Library & Tomato Leaf Analysis</div>', unsafe_allow_html=True)
