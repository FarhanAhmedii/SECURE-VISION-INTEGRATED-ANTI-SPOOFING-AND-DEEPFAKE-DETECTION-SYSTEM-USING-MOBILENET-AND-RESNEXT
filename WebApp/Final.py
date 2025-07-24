import streamlit as st
import cv2
import numpy as np
import tempfile
from inference_sdk import InferenceHTTPClient
from deepfake_detection import predict_deepfake

# Set theme to dark mode
st.markdown(
    """
    <style>
    body {
        color: white;
        background-color: #121212;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="8RSJzoEweFB7NxxNK6fg"
)

# Function to perform face anti-spoofing inference and display results
def perform_face_verification(image):
    # Perform inference on the input image
    result = CLIENT.infer(image, model_id="face-anti-spoofing-icbck/1")

    # Extract predictions from the result
    predictions = result['predictions']
    if predictions:
        # Sort predictions by confidence score in descending order
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Get the top prediction
        top_prediction = predictions[0]
        top_label = top_prediction['class']
        top_confidence = top_prediction['confidence']

        # Display the prediction result
        st.markdown(f"### Face Verification Result: {top_label.capitalize()} ({top_confidence:.2f} confidence)")

        # Display verification status based on prediction
        if top_label == 'real':
            st.success("✅ Verified!")
            return True
        else:
            st.error("❌ Spoof Detected!")
            return False
    else:
        st.write("No predictions found.")
        return False

# Function to perform deepfake detection and display results
def perform_deepfake_detection(video_path):
    # Perform deepfake detection on the input video
    prediction_result, confidence, image_path = predict_deepfake(video_path)

    # Render the template with the prediction result
    st.markdown(f"### Deepfake Detection Result: {prediction_result.capitalize()} ({confidence:.2f} confidence)")
    if prediction_result.lower() == 'fake':
        st.error("❌ Deepfake Detected!")
    else:
        st.success("✅ No Deepfake Detected!")
    st.video(video_path, format='video/mp4')

# Streamlit app code
def main():
    st.title('Secure Vision')

    # Description of the app
    st.markdown("""
    This is a face verification and deepfake detection app.
    You can verify the authenticity of a face and check if a video contains a deepfake.
    """)
    st.markdown("---")
    st.header("Choose Task")

    # Task selection
    task = st.radio("Select Task:", ("Face Verification", "Deepfake Detection"))

    if task == "Face Verification":
        st.markdown("## Face Verification")

        # Choose input method for face verification
        input_method = st.radio(
            "Choose Input Method:",
            ("Upload Image 📤", "Capture Image 📸")
        )

        if input_method == "Upload Image 📤":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Read the uploaded image
                image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                # Convert the image to grayscale for inference
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Perform face verification
                perform_face_verification(gray_image)

                # Display the uploaded image
                st.image(image, channels="BGR", caption='Uploaded Image', use_column_width=True)

        elif input_method == "Capture Image 📸":
            if st.button("Capture Image"):
                cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    st.error("Error: Could not open the camera.")
                    return

                ret, frame = cap.read()
                cap.release()

                if not ret:
                    st.error("Error: Failed to capture frame from the camera.")
                    return

                # Convert the frame to grayscale for inference
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Perform face verification
                perform_face_verification(gray_frame)

                # Display the captured image
                st.image(frame, channels="BGR", caption='Captured Image', use_column_width=True)

    elif task == "Deepfake Detection":
        st.markdown("## Deepfake Detection")

        # Upload video for deepfake detection
        uploaded_video = st.file_uploader("Upload Video for Deepfake Detection", type=["mp4"])
        if uploaded_video is not None:
            # Save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_video.read())
                video_path = temp_file.name

            # Perform deepfake detection
            perform_deepfake_detection(video_path)

if __name__ == "__main__":
    main()
