🐟 Fish Classification

A deep learning project to classify fish images into multiple species using CNNs and Transfer Learning, with a Streamlit app for real-time predictions.

⸻

📂 Project Structure

fish-classification/
│── data/            # dataset (train/val/test)
│── models/          # saved models (.keras)
│── notebooks/       # training scripts
│── streamlit_app/   # Streamlit app
│── utils/           # helper scripts
│── requirements.txt # dependencies
│── README.md        # documentation


⸻

⚙️ Setup

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/fish-classification.git](https://github.com/ioskinjal/fish-classification.git)
cd fish-classification

Create and activate a virtual environment (Python 3.10 recommended):

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt


⸻

🏋️ Train Model

Train a CNN from scratch:

python notebooks/train_cnn.py

This will save the trained model in models/cnn_from_scratch.keras.

⸻

🌐 Run Streamlit App

Start the app:

streamlit run streamlit_app/app.py

Then open http://localhost:8501 in your browser.

Upload a fish image → get species prediction + confidence score.

⸻

🚀 Future Improvements
	•	Add top-3 predictions with confidence scores
	•	Compare multiple transfer learning models (VGG16, ResNet50, etc.)
	•	Deploy app on Streamlit Cloud or Hugging Face Spaces

⸻

📜 License

This project is licensed under the MIT License.

