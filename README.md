# pothole-detection-streamlit

# 🕳️ Automated Pothole Detection System

A web-based application that automatically detects potholes in road images or videos using a pretrained **YOLOv8s model**. This tool allows road authorities, researchers, or civil engineers to monitor road conditions efficiently by identifying pothole locations and severity.

---

## Live Demo

Try the app live here: [Automated Pothole Detection](https://your-streamlit-app-link)

---

## Features

- **Image & Video Detection:** Upload road images or videos for pothole detection.  
- **Frame-by-Frame Processing:** Optimized for videos to prevent memory overflow.  
- **Download Results:** Save processed images or videos with potholes annotated.  
- **Adjustable Confidence Threshold:** Fine-tune detection sensitivity in real-time.  
- **Interactive Interface:** Built with Streamlit for easy visualization and interaction.

---

## Technologies Used

- **Python 3.10+**  
- **Streamlit** – interactive web app interface  
- **Ultralytics YOLOv8** – object detection model  
- **OpenCV** – image and video processing  
- **NumPy** – numerical computations  

---

## Installation

1. **Clone the repository:**
   ```bash
    git clone https://github.com/your-username/automated-pothole-detection.git
    cd automated-pothole-detection
2. **Create a virtual environment (recommended):**
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
3. **Install dependencies and packages:**
    pip install -r requirements.txt
    pip install -r packages.txt
4. **Download/Place the YOLOv8 model:**
    Place the trained **best.pt** file in the root directory of the project.

---

## Usage

1. **Run the Streamlit app:**
    streamlit run app.py
2. **Upload an image or video of a road through the interface.**
3. **Preview detections: The system will highlight potholes and classify severity.**
4. **Download processed results as an image or video.**

---

## Sidebar Settings

- Confidence Threshold: Adjust detection sensitivity. Lower values detect more potholes but may reduce accuracy.
- Model Info: Displays information about the YOLOv8s model and its purpose.

---

## Notes
- The app processes videos frame by frame to prevent memory overflow.
- GPU acceleration is supported if available for faster detection.

---

## Future Improvements
- Real-time camera feed detection.
- Pothole severity scoring.
- Deploy as a web service for city road monitoring.





