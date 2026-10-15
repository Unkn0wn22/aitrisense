import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = str(__import__("pathlib").Path(__file__).parent / "network_attack_detector.joblib")
META_PATH = str(__import__("pathlib").Path(__file__).parent / "network_attack_detector_metadata.json")
MEDIANS_PATH = str(__import__("pathlib").Path(__file__).parent / "network_attack_detector_medians.json")

st.set_page_config(
    page_title="AI Network Attack Detector",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_detector():
    artifact = joblib.load(MODEL_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    with open(MEDIANS_PATH, "r") as f:
        medians = json.load(f)

    return (
        artifact["model"],
        metadata["feature_names"],
        medians
    )

model, feature_names, medians = load_detector()

st.title("🛡️ AI Network Attack Detector")
st.caption("CIC-IDS2017 | Network-flow analysis")

st.warning(
    "Research prototype. Predictions are not proof of an attack, "
    "and the model has not been validated on live network traffic."
)

st.header("Analyze network flows")

uploaded_file = st.file_uploader(
    "Upload a CSV containing network-flow features",
    type=["csv"]
)

with st.expander("View required features"):
    st.code(", ".join(feature_names))

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        df.columns = df.columns.astype(str).str.strip()

        missing = [c for c in feature_names if c not in df.columns]

        if missing:
            st.error(
                f"Your file is missing {len(missing)} required features."
            )
            st.write(missing)

        elif df.empty:
            st.error("The CSV contains no records.")

        else:
            X = df[feature_names].apply(
                pd.to_numeric, errors="coerce"
            )

            X = X.replace([np.inf, -np.inf], np.nan)

            # Use medians saved from the prepared dataset
            X = X.fillna(pd.Series(medians))
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            results = df.copy()
            results["Prediction"] = np.where(
                predictions == 1, "ATTACK", "BENIGN"
            )
            results["Attack probability (%)"] = (
                probabilities * 100
            ).round(2)

            attacks = int((predictions == 1).sum())
            benign = int((predictions == 0).sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Flows analyzed", len(results))
            col2.metric("Potential attacks", attacks)
            col3.metric("Predicted benign", benign)

            if attacks:
                st.error(f"{attacks} flow(s) flagged for review.")
            else:
                st.success("No flows were flagged in this file.")

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download results CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="network_attack_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Analysis failed: {e}")

st.divider()
st.caption(
    "Educational use only. Do not rely on these predictions alone "
    "to block traffic or make security decisions."
)
