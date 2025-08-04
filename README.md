# crosswalk-violation-detection-auto
crosswalk-violation-detection-auto is a computer vision system to monitor whether pedestrians are using zebra crossings or not while crossing roads. It supports both **manual** and **automatic** zone detection using YOLO models.

## 🔧 Modes

- **automatic_mode.py**: Automatically detects zebra crossing zones using a trained YOLO model.


## 📁 Project Structure

- `models/` – YOLO models for pedestrian and zebra crossing detection
- `test_videos/` – Sample video screenshot for testing
- `main.py` – Optional script to detect the person are in zebracrossing or not with auto mode

## 🚀 How to Run

Install dependencies:
```bash
pip install -r requirements.txt
