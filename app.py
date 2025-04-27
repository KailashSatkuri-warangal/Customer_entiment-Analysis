import pandas as pd
import nltk
import streamlit as st
import plotly.express as px
import time
import os
import pickle
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure minimal logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def clean_text(text):
    """Minimal text cleaning."""
    if pd.isna(text):
        return ""
    return str(text).strip()

def analyze_sentiment(text):
    """Analyze sentiment using VADER."""
    score = sid.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive', compound
    elif compound <= -0.05:
        return 'negative', compound
    return 'neutral', compound

@st.cache_data(show_spinner=False)
def load_data(file_path, review_col, encoding='utf-8', sample_size=500):
    """Load and preprocess data with caching and sampling for memory efficiency."""
    try:
        cache_file = file_path + '.pkl'
        # Check if cache exists and is valid
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                    # Verify required columns
                    if 'sentiment' not in df.columns or 'sentiment_score' not in df.columns:
                        st.warning(f"Invalid cache for {file_path}. Attempting to regenerate...")
                        try:
                            os.remove(cache_file)  # Try to delete invalid cache
                        except PermissionError as pe:
                            st.error(
                                f"Cannot delete {cache_file}: {pe}. "
                                "Please close any programs accessing this file (e.g., file explorer, IDE) "
                                "or manually delete it and retry."
                            )
                    else:
                        return df
            except Exception as e:
                st.warning(f"Error reading cache {cache_file}: {e}. Regenerating...")

        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding, usecols=[review_col], low_memory=False)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, usecols=[review_col], engine='openpyxl')
        else:
            st.error(f"Unsupported file format: {file_path}")
            return None

        # Verify column existence
        if review_col not in df.columns:
            st.error(f"Column '{review_col}' not found in {file_path}. Available columns: {df.columns.tolist()}")
            return None

        # Sample data to reduce memory usage
        df = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df

        # Clean and analyze sentiment
        df['cleaned_review'] = df[review_col].apply(clean_text)
        sentiments = df['cleaned_review'].apply(analyze_sentiment)
        df['sentiment'] = [s[0] for s in sentiments]
        df['sentiment_score'] = [s[1] for s in sentiments]

        # Try to cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df[['cleaned_review', 'sentiment', 'sentiment_score']], f)
        except PermissionError as pe:
            print(
                f"Cannot write cache to {cache_file}: {pe}. "
                "Data loaded without caching. Please close programs accessing this file."
            )
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def compute_comparison_data(platforms_info, data_dir, selected_platforms):
    """Cache comparison data for selected platforms."""
    all_data = []
    for platform_name, path, review_col, encoding in platforms_info:
        if platform_name in selected_platforms:
            df = load_data(path, review_col, encoding)
            if df is not None:
                summary = {
                    'Platform': platform_name,
                    'Positive': (df['sentiment'] == 'positive').sum(),
                    'Negative': (df['sentiment'] == 'negative').sum(),
                    'Neutral': (df['sentiment'] == 'neutral').sum()
                }
                all_data.append(summary)
    return pd.DataFrame(all_data) if all_data else None

def plot_sentiment_distribution(df, platform_name):
    """Visualize sentiment distribution for a platform."""
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
    os.makedirs(data_dir, exist_ok=True)

    platforms_info = [
        ("BestBuy", os.path.join(data_dir, 'BestBut_Review.xlsx'), 'review_text', 'utf-8'),
        ("eBay", os.path.join(data_dir, 'ebay_reviews.csv'), 'review content', 'utf-8'),
        ("Flipkart", os.path.join(data_dir, 'flipkart_product.csv'), 'Summary', 'latin1'),
        ("Walmart", os.path.join(data_dir, 'wallmart_review.csv'), 'Review', 'utf-8')
    ]

    # Sidebar for comparison and individual platform selection
    st.sidebar.title("📊 Comparison")
    selected_platforms = st.sidebar.multiselect(
        "Select Platforms to Compare",
        [p[0] for p in platforms_info],
        default=[p[0] for p in platforms_info]  # Default to all platforms
    )
    compare_button = st.sidebar.button("Generate Comparison")

    # Comparison section at the top
    st.subheader("📊 Sentiment Count Comparison Across Companies")
    if compare_button and selected_platforms:
        with st.spinner('🔄 Generating Comparison...'):
            start_time = time.time()
            comparison_df = compute_comparison_data(platforms_info, data_dir, tuple(selected_platforms))
            if comparison_df is not None:
                fig = px.bar(
                    comparison_df.melt(id_vars='Platform', value_vars=['Positive', 'Negative', 'Neutral']),
                    x='Platform',
                    y='value',
                    color='variable',
                    barmode='group',
                    labels={'value': 'Reviews', 'variable': ''},
                    height=400,
                    template='simple_white',
                )
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                st.subheader("🧠 3D Sentiment Distribution")
                fig3d = px.scatter_3d(
                    comparison_df,
                    x='Positive',
                    y='Negative',
                    z='Neutral',
                    color='Platform',
                    size='Positive',
                    height=400,
                    template='simple_white',
                )
                fig3d.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig3d, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("No data available for selected platforms.")
            st.info(f"Comparison completed in {time.time() - start_time:.2f} seconds")

    # Individual platform analysis
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

if __name__ == "__main__":
    main()