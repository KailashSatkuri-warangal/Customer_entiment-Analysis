import pandas as pd
import nltk
import streamlit as st
import plotly.express as px
import time
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def analyze_sentiment(text):
    score = sid.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive', compound
    elif compound <= -0.05:
        return 'negative', compound
    return 'neutral', compound

def load_data(file_path, review_col, encoding='utf-8'):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            st.error(f"Unsupported file format: {file_path}")
            return None

        # Check if the required column exists
        if review_col not in df.columns:
            st.error(f"Column '{review_col}' not found in the dataset.")
            return None
        
        # Clean and analyze sentiment
        df['cleaned_review'] = df[review_col].apply(clean_text)
        sentiments = df['cleaned_review'].apply(analyze_sentiment).apply(pd.Series)
        df['sentiment'] = sentiments[0]
        df['sentiment_score'] = sentiments[1]
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

def plot_sentiment_distribution(df, platform_name):
    time.sleep(2)
    summary = {
        'positive': (df['sentiment'] == 'positive').sum(),
        'negative': (df['sentiment'] == 'negative').sum(),
        'neutral': (df['sentiment'] == 'neutral').sum()
    }
    total = sum(summary.values())
    if total > 0:
        for key in summary:
            summary[key] = summary[key] / total

    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            x=list(summary.keys()),
            y=list(summary.values()),
            color=list(summary.keys()),
            labels={'x': 'Sentiment', 'y': 'Proportion'},
            title=f"{platform_name} Sentiment - Bar Chart",
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            names=list(summary.keys()),
            values=list(summary.values()),
            title=f"{platform_name} Sentiment - Pie Chart",
            hole=0.4,
            height=500
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader(f"📈 {platform_name} Sentiment Score (3D Line Chart)")
    avg_scores = df.groupby('sentiment')['sentiment_score'].mean().reset_index()
    fig_line = px.line_3d(
        avg_scores,
        x='sentiment',
        y='sentiment_score',
        z='sentiment_score',
        markers=True,
        title=f"{platform_name} Average Sentiment Score"
    )
    st.plotly_chart(fig_line, use_container_width=True)

def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="📈", layout="wide")

    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>📊 Sentiment Analysis Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>E-commerce Platforms Review Analysis</h4>
        <hr style='border: 1px solid #4A90E2;'>
        """, unsafe_allow_html=True
    )

    data_dir = 'data'

    st.sidebar.title("📁 Platforms")
    platform = st.sidebar.selectbox("Choose a Platform", (
        "Amazon", "BestBuy", "eBay", "Flipkart", "Walmart", "Sixth"))

    if platform == "Amazon":
        st.subheader("📦 Amazon Review Analysis")
        if st.button("Generate Amazon Sentiment"):
            with st.spinner('🔄 Generating...'):
                df = load_data(os.path.join(data_dir, 'amazon_review.csv'), 'reviewText')
                if df is not None:
                    plot_sentiment_distribution(df, "Amazon")

    elif platform == "BestBuy":
        st.subheader("🛒 BestBuy Review Analysis")
        if st.button("Generate BestBuy Sentiment"):
            with st.spinner('🔄 Generating...'):
                df = load_data(os.path.join(data_dir, 'BestBut_Review.xlsx'), 'review_text')
                if df is not None:
                    plot_sentiment_distribution(df, "BestBuy")

    elif platform == "eBay":
        st.subheader("📦 eBay Review Analysis")
        if st.button("Generate eBay Sentiment"):
            with st.spinner('🔄 Generating...'):
                df = load_data(os.path.join(data_dir, 'ebay_reviews.csv'), 'review content')
                if df is not None:
                    plot_sentiment_distribution(df, "eBay")

    elif platform == "Flipkart":
        st.subheader("📱 Flipkart Review Analysis")
        if st.button("Generate Flipkart Sentiment"):
            with st.spinner('🔄 Generating...'):
                df = load_data(os.path.join(data_dir, 'flipkart_product.csv'), 'Summary', encoding='latin1')
                if df is not None:
                    plot_sentiment_distribution(df, "Flipkart")

    elif platform == "Walmart":
        st.subheader("🏬 Walmart Review Analysis")
        if st.button("Generate Walmart Sentiment"):
            with st.spinner('🔄 Generating...'):
                df = load_data(os.path.join(data_dir, 'wallmart_review.csv'), 'Review')
                if df is not None:
                    plot_sentiment_distribution(df, "Walmart")

    elif platform == "Sixth":
        st.subheader("🛍️ Sixth Platform Review Analysis")
        if st.button("Generate Sixth Sentiment"):
            with st.spinner('🔄 Generating...'):
                df = load_data(os.path.join(data_dir, 'sixth_file.csv'), 'review')
                if df is not None:
                    plot_sentiment_distribution(df, "Sixth Platform")

    st.subheader("📊 Sentiment Count Comparison Across Companies")

    platforms_info = [
        ("Amazon", os.path.join(data_dir, 'amazon_review.csv'), 'reviewText', 'utf-8'),
        ("BestBuy", os.path.join(data_dir, 'BestBut_Review.xlsx'), 'review_text', 'utf-8'),
        ("eBay", os.path.join(data_dir, 'ebay_reviews.csv'), 'review content', 'utf-8'),
        ("Flipkart", os.path.join(data_dir, 'flipkart_product.csv'), 'Summary', 'latin1'),
        ("Walmart", os.path.join(data_dir, 'wallmart_review.csv'), 'Review', 'utf-8'),
        ("Sixth", os.path.join(data_dir, 'sixth_file.csv'), 'review', 'utf-8')
    ]

    all_data = []
    for platform_name, path, review_col, encoding in platforms_info:
        df = load_data(path, review_col, encoding)
        if df is not None:
            summary = {
                'Platform': platform_name,
                'Positive': (df['sentiment'] == 'positive').sum(),
                'Negative': (df['sentiment'] == 'negative').sum(),
                'Neutral': (df['sentiment'] == 'neutral').sum()
            }
            all_data.append(summary)

    if all_data:
        comparison_df = pd.DataFrame(all_data)

        time.sleep(2)

        fig = px.bar(
            comparison_df.melt(id_vars='Platform', value_vars=['Positive', 'Negative', 'Neutral']),
            x='Platform',
            y='value',
            color='variable',
            barmode='group',
            title="Sentiment Comparison Across Platforms",
            labels={'value': 'Number of Reviews', 'variable': 'Sentiment'},
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 3D Sentiment Distribution")

        fig3d = px.scatter_3d(
            comparison_df,
            x='Positive',
            y='Negative',
            z='Neutral',
            color='Platform',
            size='Positive',
            title="3D Sentiment Distribution Across Platforms",
            height=600
        )
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
