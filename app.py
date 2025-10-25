import streamlit as st
import pandas as pd
import joblib
import pickle
import gdown
import os

# --------- CONFIG ---------
# these two files (feature_maps.pkl + this app.py) are in the GitHub repo
FEATURE_MAPS_PATH = "feature_maps.pkl"

# big model is NOT in repo, we download it from Google Drive
MODEL_PATH = "lightgbm_sales_classifier.pkl"
MODEL_ID = "1pKT2LAU-fimnEopM0QgG0e7QJaCgb7o-"  # your Drive file ID

def download_model_if_needed():
    # download the model from Google Drive only if it's not already on disk
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# --------- LOAD ARTIFACTS ---------
@st.cache_resource
def load_artifacts():
    # 1) make sure model file exists locally
    download_model_if_needed()

    # 2) load feature maps (small .pkl that lives in repo)
    feature_maps = joblib.load(FEATURE_MAPS_PATH)

    # 3) load LightGBM artifact dict
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    features = artifact["features"]
    label_mapping = artifact["label_mapping"]
    best_threshold = artifact["best_threshold"]

    return feature_maps, model, features, label_mapping, best_threshold

feature_maps, model, features, label_mapping, best_threshold = load_artifacts()

# reverse mapping for final text label (0->bad, 1->good)
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# --------- HELPER: build feature row ----------
def prepare_features(platform, genre, publisher, feature_maps, features):
    row = {
        'Platform': platform,
        'Genre': genre,
        'Publisher': publisher,
        'Publisher_avg_sales': feature_maps['publisher_avg_sales_map'].get(publisher, 0),
        'Genre_avg_sales': feature_maps['genre_avg_sales_map'].get(genre, 0),
        'Platform_avg_sales': feature_maps['platform_avg_sales_map'].get(platform, 0),
        'Platform_Genre': f"{platform}_{genre}",
        'Platform_Publisher': f"{platform}_{publisher}",
        'Genre_Publisher': f"{genre}_{publisher}",
        'Publisher_rank': feature_maps['publisher_rank_map'].get(publisher, 0),
        'Genre_rank': feature_maps['genre_rank_map'].get(genre, 0),
        'Platform_rank': feature_maps['platform_rank_map'].get(platform, 0),
    }

    df = pd.DataFrame([row])

    # ensure same dtypes as training
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    # reorder/select same columns model expects
    df = df[features]

    return df

# --------- UI LAYOUT ----------
st.title("Video Game Sales Quality Prediction")
st.write("Will this game be a GOOD seller or BAD seller?")

# dropdown options from training maps
platform_options = list(feature_maps['platform_rank_map'].keys())
genre_options = list(feature_maps['genre_rank_map'].keys())
publisher_options = list(feature_maps['publisher_rank_map'].keys())

col1, col2, col3 = st.columns(3)

with col1:
    platform = st.selectbox("Platform", sorted(platform_options))

with col2:
    genre = st.selectbox("Genre", sorted(genre_options))

with col3:
    publisher = st.selectbox("Publisher", sorted(publisher_options))

if st.button("Predict"):
    # build model input
    test_df = prepare_features(
        platform=platform,
        genre=genre,
        publisher=publisher,
        feature_maps=feature_maps,
        features=features
    )

    # LightGBM predicted prob for "good" class
    prob_good = model.predict_proba(test_df)[:, 1][0]

    # apply custom threshold from training
    pred_label_num = int(prob_good >= best_threshold)

    # map numeric -> text label ("good"/"bad")
    pred_name = inv_label_mapping[pred_label_num].upper()

    st.subheader("Result")
    st.write(f"Prediction: {pred_name}")
    st.write(f"Probability GOOD: {prob_good:.3f}")
    st.write(f"Decision Threshold: {best_threshold:.2f}")

    st.caption("Model: LightGBM classifier trained on Platform / Genre / Publisher features.")

with st.expander("Show model input row"):
    if 'test_df' in locals():
        st.dataframe(test_df)
    else:
        st.write("Click Predict first.")
