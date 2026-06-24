# 🏪 Real-Time Vision Intelligence for Autonomous Retail 
 
A production-ready retail security system that detects, tracks, and alerts on suspicious item removal using computer vision and deep learning. 
 
## 🎯 Key Features 
- YOLOv8m trained on 8185 retail shelf images (91%% mAP50 accuracy) 
- ByteTrack multi-object tracking with unique IDs 
- Virtual fence zones (green shelf zone + red exit zone) 
- Smart theft detection with audio alarm 
- PostgreSQL database for permanent event logging 
- Streamlit dashboard for store owner management 
- Cashier marks items as sold to prevent false alerts 
 
## 🛠️ Tech Stack 
- Object Detection: YOLOv8m 
- Tracking: ByteTrack 
- Dataset: SKU-110K (8185 images) 
- Database: PostgreSQL + SQLAlchemy 
- Dashboard: Streamlit + Plotly 
- Deep Learning: PyTorch 2.1.2 + CUDA 
- GPU: NVIDIA RTX 3060 Laptop 
 
## 📊 Model Performance 
- mAP50: 91.1%% 
- Precision: 90.7%% 
- Recall: 84.4%% 
- Training Images: 8185 
- Epochs: 10 
 
## 🏃 How to Run 
1. Start PostgreSQL: pg_ctl -D postgresql_data start 
2. Start detection: python fence.py 
3. Launch dashboard: streamlit run dashboard.py 
 
## 👨‍💻 Author 
Muhammad Faraan Shahid - GitHub: faraan1 
