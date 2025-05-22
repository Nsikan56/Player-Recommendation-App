<p align="center">
  <img src="Banner.png" alt="Player-Club Fit Analyzer Banner" width="100%">
</p>

# ⚽ Player-Club Fit Analyzer 
*"Moneyball meets Football Tactics" - Your personal squad optimizer*

[![Live Demo](https://img.shields.io/badge/Demo-Live%20App-brightgreen)](https://your-app-link.com) 
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![React](https://img.shields.io/badge/React-18-purple)](https://reactjs.org)

## 🎯 What It Does
A **machine learning-powered** tool that helps football clubs:
- 🔍 **Find perfect player matches** using K-Means clustering
- 💰 **Optimize transfers** with financial ROI analysis
- 📊 **Visualize gaps** in squad composition

Built for my MSc Final Project with real-world football data sourced from FBref and Transfermarkt

## 🛠️ Tech Stack
| Area       | Technologies Used |
|------------|-------------------|
| **Backend** | Python, Flask, Scikit-learn, Pandas |
| **Frontend**| React.js, Tailwind CSS |
| **Data**    | Transfermarkt + FBref datasets |

## 📂 Project Structure
```bash
.
├── backend/          # The brains: ML models & API
│   ├── app.py        # Flask endpoints
│   ├── player_analysis.py  # Magic happens here
│   └── data.csv      # 3,500+ player records
│
└── frontend/         # Sleek dashboard
    ├── components/   # Interactive visuals
    └── lib/          # Data processing logic
