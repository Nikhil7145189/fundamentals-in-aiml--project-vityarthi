# 📊 Study Room Occupancy Tracker & Predictor

## 🎯 Project Overview
This project is a Command Line Interface (CLI) application designed to solve a common campus problem: finding an available study room. It uses a two-part Machine Learning pipeline to automatically scan images of a room, count the number of people, and use historical data to predict future occupancy.

This was built as a Bring Your Own Project (BYOP) capstone, applying core concepts from **Fundamentals of AI and ML**.

## 🧠 Course Concepts Applied
* **Module 5 (Transfer Learning & Pre-trained Models):** Utilizes OpenCV's Haar Cascades (a pre-trained Computer Vision model) to detect human faces without requiring massive computational training.
* **Module 4 (Linear Regression & Machine Learning):** Uses `scikit-learn` to train a Linear Regression model on generated CSV data, predicting future room occupancy based on time-of-day trends.
* **Module 3 (Feature Learning):** Converts string-based timestamps into continuous decimal features so the regression model can process the math.

## 📁 Repository Structure
```
study_room_tracker/
│
├── test_images/            # Folder containing raw images of the room (Face dataset)
│
├── tracker.py              # Script 1: The Vision Engine (Data Collector)
├── analyzer.py             # Script 2: The ML Predictor (Data Analyzer)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
⚙️ Installation & Setup
Clone this repository to your local machine.

Ensure you have Python 3 installed.

Install the required dependencies by running the following command in your terminal:
pip install -r requirements.txt

Place your testing images (wide shots of crowds/rooms) into the test_images directory.

🚀 How to Use
Part 1: Gather Data (tracker.py)
This script acts as the automated sensor. It scans a folder of images, counts the faces using AI, and silently logs the data to a file named occupancy_log.csv. Note: This script includes an automated time-simulator that increments the clock by 45 minutes for each image to build a realistic daily dataset.

Command:
python tracker.py --dir test_images --capacity 20

--dir: The folder containing your images.
--capacity: The maximum number of seats in the room.

Part 2: Predict the Future (analyzer.py)
Once you have generated data using the tracker, this script reads the CSV file, trains a Linear Regression model in real-time, and predicts how full the room will be at a future time.

Command:
python analyzer.py --time 14:30 --capacity 20

--time: The future time you want to check (in HH:MM format).
--capacity: The maximum number of seats in the room.

🔮 Future Scope & Scalability
While this MVP uses localized Haar Cascades for indoor study rooms, the architecture is designed to scale. By upgrading the vision model to YOLO (Deep Learning) and replacing simple Linear Regression with Multiple Regression (incorporating weather, day of the week, and holiday data), this system could easily be scaled to map the real-time foot traffic and density of an entire city block.
