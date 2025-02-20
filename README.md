# Plant Disease Detection App

## Overview
This project is an AI-powered **Plant Disease Detection** web application using a **Flask API** and a **ReactJS frontend**. The backend runs a **TensorFlow Lite** model to classify plant diseases from uploaded images. The app is designed for **farmers and researchers** to quickly diagnose plant diseases and take preventive measures.

## Features
- Upload plant leaf images for disease detection.
- AI-based prediction using a TensorFlow Lite model.
- Flask backend API for processing requests.
- ReactJS frontend for user interaction.
- Future plans: Mobile app integration and cloud deployment.

---

## Tech Stack
### Backend:
- **Flask** (Python-based API)
- **TensorFlow Lite** (Machine Learning Model)
- **Google Drive** (Model Hosting)
- **Flask-CORS** (Handling frontend-backend communication)

### Frontend:
- **ReactJS** (User Interface)
- **Axios** (Making API requests)
- **Tailwind CSS** (Styling)

---

## Installation and Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-repo/plant-disease-detection.git
cd plant-disease-detection
```

### 2️⃣ Setup the Backend (Flask)
#### Install Dependencies
```sh
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

#### Run the Flask API
```sh
python app.py
```
- The server will start at `http://127.0.0.1:5000`

---

### 3️⃣ Setup the Frontend (ReactJS)
#### Install Dependencies
```sh
cd frontend
npm install
```

#### Run the React App
```sh
npm start
```
- The React app will open at `http://localhost:3000`

---

## API Endpoints
### **POST /predict** (Plant Disease Prediction)
**Request:**
- `image`: Upload an image file of a plant leaf.

**Response:**
```json
{
  "prediction": "Tomato Leaf Blight"
}
```

---

## Troubleshooting
### ❌ API Not Working?
- Ensure Flask is running (`python app.py`).
- Check API logs for errors.
- Test using `cURL`:
  ```sh
  curl -X POST -F "image=@test.jpg" http://127.0.0.1:5000/predict
  ```

### ❌ React Not Connecting to API?
- Update `API_URL` in React:
  ```js
  const API_URL = "http://127.0.0.1:5000/predict";
  ```
- Check CORS errors (Install `Flask-CORS` in backend).

---

## Future Enhancements
- 🌍 Deploy Flask API to **Google Cloud Run**
- 📱 Convert ReactJS frontend into a **mobile app** (React Native)
- 💰 Monetization through **premium disease analysis reports**

---

## Contributors
- **Your Name** - [GitHub Profile](https://github.com/your-github)

**Happy Coding! 🚀**

