# 📊 Customer Sentiment Analysis Dashboard  

An interactive **Sentiment Analysis Dashboard** built with **Streamlit**, **NLTK VADER**, and **Plotly** for analyzing customer reviews across multiple **E-commerce platforms** (Amazon, BestBuy, eBay, Flipkart, Walmart, and Sixth Platform).  

This project helps businesses **understand customer opinions** from product reviews by classifying them into **positive, negative, and neutral sentiments** with visual insights.  
---
## Live
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reviewvibe.streamlit.app)

---

## 🚀 Features  

- ✅ Supports multiple e-commerce platforms (Amazon, BestBuy, eBay, Flipkart, Walmart, etc.)  
- ✅ Sentiment classification using **NLTK VADER**  
- ✅ Data preprocessing and cleaning with **Pandas**  
- ✅ Interactive visualizations using **Plotly Express**  
- ✅ Platform-wise sentiment distribution (Bar chart, Pie chart, 3D Line chart)  
- ✅ Cross-platform sentiment comparison with **3D scatter plots**  
- ✅ Caching for optimized performance  

---

## 📂 Project Structure  

```
📦 Sentiment-Analysis-Dashboard
├── app.py              # Main Streamlit app
├── app_flask.py        # Flask-based backend (optional)
├── analyzer.py         # Sentiment analyzer logic
├── preprocess.py       # Data preprocessing scripts
├── requirements.txt    # Python dependencies
├── data/               # Folder containing datasets
│   ├── amazon_review.csv
│   ├── BestBut_Review.xlsx
│   ├── ebay_reviews.csv
│   ├── flipkart_product.csv
│   ├── wallmart_review.csv
│   └── sixth_file.csv
└── README.md           # Project Documentation
```

---

## 📊 Datasets  

The project uses **customer review datasets** from different e-commerce platforms:  

- `amazon_review.csv` → Amazon reviews  
- `BestBut_Review.xlsx` → BestBuy reviews  
- `ebay_reviews.csv` → eBay reviews  
- `flipkart_product.csv` → Flipkart reviews  
- `wallmart_review.csv` → Walmart reviews  
- `sixth_file.csv` → Sixth Platform reviews  

---

## ⚙️ Installation  

1. Clone the repository  

```bash
git clone https://github.com/KailashSatkuri-warangal/Sentiment-Analysis-Dashboard.git
cd Sentiment-Analysis-Dashboard
```

2. Create and activate a virtual environment  

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies  

```bash
pip install -r requirements.txt
```

