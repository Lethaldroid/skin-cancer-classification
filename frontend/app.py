import streamlit as st
import requests
from PIL import Image

# --- CONFIGURATION ---
# Local model inference service
API_URL = "http://localhost:8000/predict"

# --- PAGE SETUP ---
st.set_page_config(page_title="Skin Diagnosis Client", page_icon="🏥")

st.title("🏥 Skin Diagnosis Client")
st.markdown("""
This tool allows you to upload a skin image.
It sends the image to your **Local AI Model** for inference.
""")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload a Skin Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Show the image preview
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Skin Image', width=300)

    # 2. Action Button
    if st.button("🚀 Analyze"):

        with st.spinner("Sending image for inference..."):
            try:
                # Reset the file pointer to the beginning before sending
                uploaded_file.seek(0)

                # Prepare the payload (simulating Postman form-data)
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                }

                # Send POST request to localhost:8000
                response = requests.post(API_URL, files=files)

                # --- RESULT HANDLING ---
                if response.status_code == 200:
                    data = response.json()

                    st.success("✅ Inference completed!")

                    prediction = data.get("prediction", "Unknown")
                    confidence = data.get("confidence", 0.0)
                    classes = data.get("classes", [])

                    # Display Metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", prediction)
                    col2.metric("Confidence", f"{confidence:.2%}")

                    if classes:
                        st.caption(f"Classes: {', '.join(classes)}")

                    # Show raw JSON for debugging
                    with st.expander("View Raw JSON Response"):
                        st.json(data)

                else:
                    st.error(f"❌ Server Error ({response.status_code}):")
                    st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Refused!")
                st.warning(f"Could not connect to {API_URL}.")
                st.info("💡 Tip: Is your model service running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {e}")