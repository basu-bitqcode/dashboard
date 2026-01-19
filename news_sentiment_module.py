import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import re
import plotly.graph_objects as go
from textblob import TextBlob
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings('ignore')

class NewsSentimentAnalyzer:
    def __init__(self, google_sheet_url=None):
        """
        Initialize the News Sentiment Analyzer
        
        Args:
            google_sheet_url: URL of the Google Sheet
        """
        self.google_sheet_url = google_sheet_url
        self.df = None

        self.financial_events = {
            "earnings_beat": {
                "patterns": ["beat estimates", "earnings beat", "profit jumps"],
                "impact": +12
            },
            "earnings_miss": {
                "patterns": ["miss estimates", "earnings miss"],
                "impact": -15
            },
            "guidance_up": {
                "patterns": ["raises guidance", "outlook raised"],
                "impact": +10
            },
            "guidance_down": {
                "patterns": ["cuts guidance", "outlook lowered"],
                "impact": -12
            },
            "rate_cut": {
                "patterns": ["rate cut", "interest rates lowered"],
                "impact": +8
            },
            "rate_hike": {
                "patterns": ["rate hike", "interest rates raised"],
                "impact": -10
            },
            "regulatory_risk": {
                "patterns": ["probe", "regulatory action", "antitrust"],
                "impact": -14
            },
            "order_win": {
                "patterns": ["order win", "new contract", "large deal"],
                "impact": +9
            },
        }

        def has_negation_or_contrast(self, text):
            negations = ["but", "however", "despite", "although", "while"]
            return any(n in text for n in negations)


        
        # Financial keywords for sentiment boosting
        self.positive_keywords = [
            'beat', 'surge', 'jump', 'rise', 'gain', 'rally', 'bull', 'positive',
            'growth', 'profit', 'increase', 'higher', 'record', 'win', 'success',
            'strong', 'optimistic', 'boom', 'breakthrough', 'dividend', 'buyback',
            'upgrade', 'outperform', 'bullish', 'recovery', 'soar', 'lifeline',
            'cut rates', 'break', 'breakout', 'outperform', 'beat estimates'
        ]
        
        self.negative_keywords = [
            'cut', 'plunge', 'drop', 'fall', 'loss', 'crash', 'bear', 'negative',
            'decline', 'decrease', 'lower', 'miss', 'fail', 'weak', 'pessimistic',
            'slump', 'downgrade', 'underperform', 'bearish', 'recession', 'warn',
            'risk', 'volatility', 'uncertainty', 'selloff', 'downturn', 'bankrupt',
            'soaring', 'beating', 'crisis', 'tumble', 'plummet', 'collapse', 'slump'
        ]
    
    def clean_text(self, text):
        """Remove emojis, special characters, and clean text for NLP"""
        if pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove Twitter handles
        text = re.sub(r'@\w+', '', text)
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove RT (retweet) mentions
        text = re.sub(r'\bRT\b', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    

    def parse_datetime(self, datetime_str):
        """
        Parse datetime string and convert to US Market Time (NASDAQ / NYSE)
        """
    
        if pd.isna(datetime_str):
            return None
    
        datetime_str = str(datetime_str).strip()
    
        formats = [
            "%B %d, %Y at %I:%M%p",
            "%B %d, %Y %I:%M%p",
            "%B %d, %Y at %I:%M:%S%p",
            "%Y-%m-%d %H:%M:%S"
        ]
    
        dt = None
        for fmt in formats:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                break
            except ValueError:
                continue
    
        if dt is None:
            return None
    
        # Assume input is UTC unless specified
        dt_utc = dt.replace(tzinfo=timezone.utc)
    
        # Convert to US Market Time (DST safe)
        dt_us = dt_utc.astimezone(ZoneInfo("America/New_York"))
    
        return dt_us

    
    def load_news_data(self):
        """Load news data from Google Sheet"""
        try:
            if self.google_sheet_url:
                # For public Google Sheets
                if '/edit#' in self.google_sheet_url:
                    # Extract sheet ID
                    if 'gid=' in self.google_sheet_url:
                        sheet_id = self.google_sheet_url.split('gid=')[1]
                        csv_url = self.google_sheet_url.replace('/edit#gid=', f'/export?format=csv&gid={sheet_id}')
                    else:
                        csv_url = self.google_sheet_url.replace('/edit#', '/export?format=csv&gid=0')
                else:
                    # Assume it's already a CSV export URL
                    csv_url = self.google_sheet_url
                
                # Make sure it's a CSV export URL
                if 'export?format=csv' not in csv_url:
                    if '/edit?' in csv_url:
                        csv_url = csv_url.replace('/edit?', '/export?format=csv&')
                    else:
                        csv_url = f"{csv_url}/export?format=csv"
                
                self.df = pd.read_csv(csv_url)
                
                if self.df is not None and not self.df.empty:
                    # Clean column names
                    self.df.columns = [col.strip() for col in self.df.columns]
                    
                    # Clean news text
                    if 'News' in self.df.columns:
                        self.df['Cleaned_News'] = self.df['News'].apply(self.clean_text)
                        # Remove empty news
                        self.df = self.df[self.df['Cleaned_News'].str.strip() != '']
                    
                    # Parse datetime
                    if 'DateTime' in self.df.columns:
                        self.df['DateTime_ET'] = self.df['DateTime'].apply(self.parse_datetime)
                    
                        self.df['Date'] = pd.to_datetime(self.df['DateTime_ET'], errors="coerce").dt.date

                        # 🔥 Drop rows where DateTime parsing failed
                        self.df = self.df.dropna(subset=['Date'])
                        
                        # Now this is SAFE
                        latest_date = self.df['Date'].max()

                    
                        # Filter only that date
                        self.df = self.df[self.df['Date'] == latest_date]
                    
                        # Sort by datetime (newest first)
                        self.df = self.df.sort_values('DateTime_ET', ascending=False)

                    
                    return True
            return False
            
        except Exception as e:
            st.error(f"Error loading news data: {str(e)}")
            return False
    
    def analyze_sentiment(self, text):
        if not text or pd.isna(text):
            return {'score': 50, 'sentiment': 'Neutral'}
    
        text_lower = text.lower()
        score = 50  # Start neutral
    
        # ---------- FINANCIAL EVENT IMPACT ----------
        for event, cfg in self.financial_events.items():
            for pattern in cfg["patterns"]:
                if pattern in text_lower:
                    score += cfg["impact"]
    
        # ---------- NEGATION / CONTRAST ----------
        if self.has_negation_or_contrast(text_lower):
            score = 50 + (score - 50) * 0.6  # dampen conviction
    
        # ---------- FALLBACK NLP (LIGHT WEIGHT) ----------
        polarity = TextBlob(text).sentiment.polarity
        score += polarity * 15  # small influence only
    
        # ---------- CLAMP ----------
        score = max(0, min(100, score))
    
        # ---------- LABEL ----------
        if score > 58:
            sentiment = "Positive"
        elif score < 42:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
    
        return {
            'score': score,
            'sentiment': sentiment,
            'polarity': polarity
        }


    
    def create_speedometer(self, sentiment_score, sentiment_label, width=420):
        """
        Clean, compact, non-overlapping speedometer
        - Wrapped tick labels
        - No redundant title
        """
    
        # ---------- COLOR ----------
        if sentiment_label == "Positive":
            main_color = "#2563EB"
        elif sentiment_label == "Negative":
            main_color = "#DC2626"
        else:
            main_color = "#6B7280"
    
        # ---------- DYNAMIC FONT SIZES ----------
        number_font = int(width * 0.13)
        label_font  = int(width * 0.055)
        tick_font   = int(width * 0.028)
    
        number_font = min(max(number_font, 28), 68)
    
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=sentiment_score,
            domain={"x": [0, 1], "y": [0.38, 1]},  # Top zone
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    # 🔥 WRAPPED LABELS
                    "ticktext": [
                        "Very<br>Bearish",
                        "Bearish",
                        "Neutral",
                        "Bullish",
                        "Very<br>Bullish"
                    ],
                    "tickfont": {
                        "size": tick_font,
                        "color": "#374151"
                    }
                },
                "bar": {
                    "color": main_color,
                    "thickness": 0.32
                },
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 40], "color": "#FEE2E2"},
                    {"range": [40, 60], "color": "#E5E7EB"},
                    {"range": [60, 100], "color": "#DBEAFE"}
                ]
            }
        ))
    
        # ---------- BIG NUMBER ----------
        fig.add_annotation(
            x=0.5,
            y=0.23,
            text=f"<b>{sentiment_score:.0f}%</b>",
            showarrow=False,
            font=dict(size=number_font, color=main_color),
            xref="paper",
            yref="paper"
        )
    
        # ---------- SENTIMENT LABEL ----------
        fig.add_annotation(
            x=0.5,
            y=0.13,
            text=f"<b>{sentiment_label.upper()}</b>",
            showarrow=False,
            font=dict(size=label_font, color=main_color),
            xref="paper",
            yref="paper"
        )
    
        fig.update_layout(
            height=int(width * 1.0),
            margin=dict(l=30, r=30, t=40, b=60),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
    
        return fig




    
    def calculate_overall_sentiment(self):
        """Calculate overall sentiment from all news"""
        if self.df is None or self.df.empty or 'Cleaned_News' not in self.df.columns:
            return {'score': 50, 'sentiment': 'Neutral', 'color': '#F59E0B', 'indicator': '●'}
        
        # Calculate weighted sentiment (recent news more important)
        if len(self.df) > 0:
            sentiments = []
            weights = []
            
            for i, (idx, row) in enumerate(self.df.iterrows()):
                if pd.notna(row['Cleaned_News']) and str(row['Cleaned_News']).strip():
                    sentiment = self.analyze_sentiment(row['Cleaned_News'])
                    # Weight: recent news gets higher weight
                    weight = 1.0 / (i + 1)  # Linear decay
                    sentiments.append(sentiment['score'])
                    weights.append(weight)
            
            if sentiments:
                weighted_avg = np.average(sentiments, weights=weights)
                
                if weighted_avg > 55:
                    sentiment_label = "Positive"
                    color = "#10B981"
                elif weighted_avg < 45:
                    sentiment_label = "Negative"
                    color = "#EF4444"
                else:
                    sentiment_label = "Neutral"
                    color = "#F59E0B"
                
                return {
                    'score': weighted_avg,
                    'sentiment': sentiment_label,
                    'color': color,
                    'indicator': '▲' if sentiment_label == "Positive" else '▼' if sentiment_label == "Negative" else '●'
                }
        
        return {'score': 50, 'sentiment': 'Neutral', 'color': '#F59E0B', 'indicator': '●'}
    
    def display_dashboard(self):
        """Display the complete news sentiment dashboard"""
        # Load data
        with st.spinner("📥 Loading news data..."):
            if self.load_news_data():
                if self.df is None and self.df.empty:
                    st.info("📭 No news available for today.")
                    return
            else:
                st.error("❌ Failed to load news data")
                return
        
        # Calculate overall sentiment
        overall_sentiment = self.calculate_overall_sentiment()
        
        # Create two columns layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Speedometer section
            st.subheader("📊 Market Sentiment Indicator")
            
            # Display speedometer
            fig = self.create_speedometer(
                overall_sentiment['score'],
                overall_sentiment['sentiment'],
                width=st.session_state.get("plot_width", 380)
                )
        
            st.plotly_chart(fig, use_container_width=True)

            
            # Sentiment statistics
            st.markdown("---")
            st.subheader("📈 Sentiment Breakdown")
            
            if not self.df.empty:
                # Calculate sentiment distribution
                sentiments = []
                for news in self.df['Cleaned_News']:
                    if pd.notna(news) and str(news).strip():
                        sentiment = self.analyze_sentiment(news)
                        sentiments.append(sentiment['sentiment'])
                
                if sentiments:
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    # Display sentiment counts with colors
                    sentiment_data = []
                    for sentiment_type, bg_color, icon in [
                        ('Positive', 'rgba(16, 185, 129, 0.1)', '🟢'),
                        ('Neutral', 'rgba(245, 158, 11, 0.1)', '🟡'),
                        ('Negative', 'rgba(239, 68, 68, 0.1)', '🔴')
                    ]:
                        count = sentiment_counts.get(sentiment_type, 0)
                        percentage = (count / len(sentiments)) * 100 if len(sentiments) > 0 else 0
                        
                        text_color = "#10B981" if sentiment_type == "Positive" else "#EF4444" if sentiment_type == "Negative" else "#F59E0B"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding: 8px; background-color: {bg_color}; border-radius: 6px;">
                            <span style="font-size: 16px; color: #1E293B;">
                                {icon} <strong>{sentiment_type}</strong>
                            </span>
                            <div style="text-align: right;">
                                <div style="font-size: 18px; font-weight: bold; color: {text_color};">
                                    {count}
                                </div>
                                <div style="font-size: 10px; color: #64748B;">
                                    {percentage:.1f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Latest update time
                st.markdown("---")
                if not self.df.empty and 'DateTime_ET' in self.df.columns:
                    latest_news_time = self.df['DateTime_ET'].iloc[0]
                    st.caption(f"""
                    <div style="text-align: center; color: #64748B; font-size: 12px;">
                        📅 Last update: {latest_news_time.strftime('%Y-%m-%d %H:%M:%S %Z')}

                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # News feed section
            st.subheader("📰 Live News Feed")
            
            # Create clean news display
            if self.df is not None and not self.df.empty:
                # Limit to 10 latest items
                display_df = self.df.head(10)
                
                # Create a container for news
                news_container = st.container()
                
                with news_container:
                    for idx, row in display_df.iterrows():
                        timestamp = row['DateTime_ET'].strftime("%H:%M:%S") if 'DateTime_ET' in row else "N/A"
                        news_text = row['Cleaned_News']
                        
                        if not news_text or str(news_text).strip() == "":
                            continue
                        
                        # Analyze sentiment
                        news_sentiment = self.analyze_sentiment(news_text)
                        
                        # Get color and indicator
                        if news_sentiment['sentiment'] == "Positive":
                            text_color = "#10B981"  # Green
                            indicator = "▲"
                        elif news_sentiment['sentiment'] == "Negative":
                            text_color = "#EF4444"  # Red
                            indicator = "▼"
                        else:
                            text_color = "#F59E0B"  # Yellow/Amber
                            indicator = "●"
                        
                        # Display the news item
                        st.markdown(
                            f"**[{timestamp}] {indicator}** <span style='color: {text_color}'>{news_text}</span>",
                            unsafe_allow_html=True
                        )
                
                # News statistics below the news items
                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("📊 Total News", len(self.df))
                with col_b:
                    if not self.df.empty and 'DateTime_ET' in self.df.columns:
                        latest_time = self.df['DateTime_ET'].iloc[0].strftime("%H:%M ET")
                        st.metric("🕒 Latest", latest_time)
                with col_c:
                    st.metric("📈 Sentiment", f"{overall_sentiment['score']:.1f}%")
            else:
                st.info("No news items to display.")
