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
        self.latest_available_date = None
        
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
        """Load news data from Google Sheet using latest available date"""
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
                        
                        # Filter for LATEST AVAILABLE DATE instead of today
                        self.df['Date'] = pd.to_datetime(self.df['DateTime_ET']).dt.date
                        
                        # Get the latest date from the data
                        self.latest_available_date = self.df['Date'].max()
                        
                        # Filter for the latest available date
                        self.df = self.df[self.df['Date'] == self.latest_available_date]
                        
                        if len(self.df) == 0:
                            # If no data for latest date, use the most recent available data
                            self.df = self.df.sort_values('Date', ascending=False).head(20)
                            if len(self.df) > 0:
                                self.latest_available_date = self.df['Date'].iloc[0]
                        
                        # Sort by datetime (newest first)
                        self.df = self.df.sort_values('DateTime_ET', ascending=False)
                    
                    return True
            return False
            
        except Exception as e:
            st.error(f"Error loading news data: {str(e)}")
            return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob with financial keyword boosting"""
        if not text or pd.isna(text):
            return {'score': 50, 'sentiment': 'Neutral', 'color': '#FFFF00', 'indicator': '●'}
        
        text_lower = str(text).lower()
        
        # Base sentiment from TextBlob
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        
        # Adjust based on financial keywords
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # Boost polarity based on keyword matches
        keyword_boost = 0
        if positive_count > negative_count:
            keyword_boost = 0.25  # Boost for positive keywords
        elif negative_count > positive_count:
            keyword_boost = -0.25  # Reduce for negative keywords
        
        # Apply keyword boost
        polarity = max(-1.0, min(1.0, polarity + keyword_boost))
        
        # Convert to score between 0 and 100
        score = (polarity + 1) * 50
        
        # Adjust thresholds for better distribution
        if score > 55:  # More sensitive positive threshold
            sentiment = "Positive"
            color = "#00FF00"  # Bright green for terminal
            indicator = "▲"
        elif score < 45:  # More sensitive negative threshold
            sentiment = "Negative"
            color = "#FF0000"  # Bright red for terminal
            indicator = "▼"
        else:
            sentiment = "Neutral"
            color = "#FFFF00"  # Yellow for terminal
            indicator = "●"
        
        return {
            'score': min(max(score, 0), 100),
            'sentiment': sentiment,
            'color': color,
            'indicator': indicator,
            'polarity': polarity,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count
        }

    
    def create_speedometer(self, sentiment_score, sentiment_label):
        """
        Enhanced professional sentiment speedometer with better styling
        """
    
        # Define colors based on sentiment
        if sentiment_label == "Positive":
            main_color = "#10B981"   # Emerald Green
            gauge_colors = ["#059669", "#10B981", "#34D399"]  # Green gradient
        elif sentiment_label == "Negative":
            main_color = "#EF4444"   # Red
            gauge_colors = ["#DC2626", "#EF4444", "#F87171"]  # Red gradient
        else:
            main_color = "#F59E0B"   # Amber
            gauge_colors = ["#D97706", "#F59E0B", "#FBBF24"]  # Amber gradient
    
        # Create gauge figure
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            number={
                "font": {
                    "size": 72,  # Larger font for better visibility
                    "color": main_color,
                    "family": "Inter, Arial, sans-serif",
                    "weight": "bold"
                },
                "suffix": "%",
                "prefix": "",
            },
            delta={
                "reference": 50,
                "position": "bottom",
                "font": {"size": 20, "family": "Inter, Arial"},
                "increasing": {"symbol": "▲", "color": "#10B981"},
                "decreasing": {"symbol": "▼", "color": "#EF4444"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticktext": [
                        "Extreme<br>Bearish",
                        "Bearish",
                        "Neutral",
                        "Bullish",
                        "Extreme<br>Bullish",
                        ""
                    ],
                    "tickfont": {
                        "size": 14,
                        "color": "#475569",
                        "family": "Inter, Arial"
                    },
                    "tickangle": 0,
                    "tickwidth": 2,
                    "tickcolor": "#CBD5E1"
                },
                "bar": {
                    "color": main_color,
                    "thickness": 0.5,
                    "line": {"color": "#1E293B", "width": 1}
                },
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "borderwidth": 2,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [40, 60], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [60, 100], "color": "rgba(16, 185, 129, 0.15)"}
                ],
                "threshold": {
                    "line": {"color": main_color, "width": 4},
                    "thickness": 0.75,
                    "value": sentiment_score
                },
                "shape": "angular"
            }
        ))
    
        # Add sentiment label with better styling
        fig.add_annotation(
            x=0.5,
            y=0.25,
            text=f"<span style='font-size:32px; font-weight:bold; color:{main_color};'>{sentiment_label.upper()}</span>",
            showarrow=False,
            font=dict(
                size=24,
                family="Inter, Arial, sans-serif"
            ),
            xref="paper",
            yref="paper"
        )
    
        # Add date information if available
        if hasattr(self, 'latest_available_date') and self.latest_available_date:
            date_str = self.latest_available_date.strftime("%Y-%m-%d")
            fig.add_annotation(
                x=0.5,
                y=-0.1,
                text=f"<span style='font-size:14px; color:#64748B;'>Latest Data: {date_str}</span>",
                showarrow=False,
                font=dict(
                    size=12,
                    family="Inter, Arial"
                ),
                xref="paper",
                yref="paper"
            )
    
        # Enhanced layout
        fig.update_layout(
            height=480,
            margin=dict(l=40, r=40, t=120, b=80),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Inter, Arial, sans-serif"),
            title={
                "text": "<b>🎯 MARKET SENTIMENT GAUGE</b>",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {
                    "size": 28,
                    "color": "#0F172A",
                    "family": "Inter, Arial, sans-serif"
                }
            }
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
        """Display the enhanced news sentiment dashboard"""
        # Load data
        with st.spinner("📥 Loading news data..."):
            if self.load_news_data():
                if self.df is None or self.df.empty:
                    st.info("📭 No news available.")
                    return
            else:
                st.error("❌ Failed to load news data")
                return
        
        # Calculate overall sentiment
        overall_sentiment = self.calculate_overall_sentiment()
        
        # Dashboard header with date info
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin-bottom: 20px;
                    color: white;">
            <h1 style="margin: 0; font-size: 28px;">📰 Market News Sentiment Dashboard</h1>
            <p style="margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;">
                Latest Available Date: {self.latest_available_date.strftime('%B %d, %Y') if self.latest_available_date else 'N/A'} 
                | Total News: {len(self.df)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Speedometer section with better styling
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 10px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        margin-bottom: 20px;">
                <h3 style="margin: 0 0 15px 0; color: #1E293B;">📊 Sentiment Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display enhanced speedometer
            fig = self.create_speedometer(
                overall_sentiment['score'],
                overall_sentiment['sentiment']
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Sentiment statistics with better styling
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 10px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        margin-top: 20px;">
                <h4 style="margin: 0 0 15px 0; color: #1E293B;">📈 Sentiment Breakdown</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if not self.df.empty:
                # Calculate sentiment distribution
                sentiments = []
                for news in self.df['Cleaned_News']:
                    if pd.notna(news) and str(news).strip():
                        sentiment = self.analyze_sentiment(news)
                        sentiments.append(sentiment['sentiment'])
                
                if sentiments:
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    # Display sentiment counts with better styling
                    sentiment_data = []
                    for sentiment_type, bg_color, icon, text_color in [
                        ('Positive', 'rgba(16, 185, 129, 0.15)', '📈', '#10B981'),
                        ('Neutral', 'rgba(245, 158, 11, 0.15)', '📊', '#F59E0B'),
                        ('Negative', 'rgba(239, 68, 68, 0.15)', '📉', '#EF4444')
                    ]:
                        count = sentiment_counts.get(sentiment_type, 0)
                        percentage = (count / len(sentiments)) * 100 if len(sentiments) > 0 else 0
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; 
                                    margin-bottom: 12px; padding: 12px; 
                                    background: linear-gradient(135deg, {bg_color}, rgba(255, 255, 255, 0.3));
                                    border-radius: 8px; border-left: 4px solid {text_color};">
                            <div style="display: flex; align-items: center;">
                                <span style="font-size: 20px; margin-right: 10px;">{icon}</span>
                                <span style="font-size: 16px; color: #1E293B; font-weight: 500;">
                                    {sentiment_type}
                                </span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 22px; font-weight: bold; color: {text_color};">
                                    {count}
                                </div>
                                <div style="font-size: 14px; color: #64748B; font-weight: 500;">
                                    {percentage:.1f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Latest update time with better styling
                st.markdown("---")
                if not self.df.empty and 'DateTime_ET' in self.df.columns:
                    latest_news_time = self.df['DateTime_ET'].iloc[0]
                    st.markdown(f"""
                    <div style="background: #F8FAFC; padding: 12px; border-radius: 8px; text-align: center;">
                        <div style="color: #64748B; font-size: 14px; font-weight: 500; margin-bottom: 5px;">
                            ⏰ Last News Update
                        </div>
                        <div style="color: #1E293B; font-size: 16px; font-weight: 600;">
                            {latest_news_time.strftime('%Y-%m-%d %H:%M:%S ET')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # News feed section with better styling
            st.markdown("""
            <div style="background: white; padding: 20px; border-radius: 10px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #1E293B;">📰 Live News Feed</h3>
                    <div style="display: flex; gap: 10px;">
                        <span style="background: #F1F5F9; padding: 5px 10px; border-radius: 6px; 
                                    color: #64748B; font-size: 14px; font-weight: 500;">
                            {len(self.df)} Items
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
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
                            bg_color = "rgba(16, 185, 129, 0.1)"
                            border_color = "#10B981"
                            text_color = "#065F46"
                            indicator = "📈"
                        elif news_sentiment['sentiment'] == "Negative":
                            bg_color = "rgba(239, 68, 68, 0.1)"
                            border_color = "#EF4444"
                            text_color = "#991B1B"
                            indicator = "📉"
                        else:
                            bg_color = "rgba(245, 158, 11, 0.1)"
                            border_color = "#F59E0B"
                            text_color = "#92400E"
                            indicator = "📊"
                        
                        # Display the news item with better styling
                        st.markdown(f"""
                        <div style="background: {bg_color}; 
                                    border-left: 4px solid {border_color};
                                    padding: 16px; 
                                    margin-bottom: 12px; 
                                    border-radius: 8px;
                                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="display: flex; justify-content: space-between; 
                                        align-items: flex-start; margin-bottom: 8px;">
                                <span style="background: {border_color}; color: white; 
                                            padding: 4px 8px; border-radius: 4px; 
                                            font-size: 12px; font-weight: bold;">
                                    {indicator} {news_sentiment['sentiment']}
                                </span>
                                <span style="color: #64748B; font-size: 14px; font-weight: 500;">
                                    ⏰ {timestamp} ET
                                </span>
                            </div>
                            <div style="color: {text_color}; font-size: 15px; line-height: 1.5;">
                                {news_text}
                            </div>
                            <div style="margin-top: 8px; color: #94A3B8; font-size: 13px;">
                                Sentiment Score: <span style="color: {border_color}; font-weight: bold;">
                                {news_sentiment['score']:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # News statistics with better styling
                st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px; 
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                            margin-top: 20px;">
                    <h4 style="margin: 0 0 15px 0; color: #1E293B;">📊 News Statistics</h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                """, unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #F8FAFC, #F1F5F9); 
                                padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 14px; color: #64748B; margin-bottom: 5px;">
                            📊 Total News
                        </div>
                        <div style="font-size: 28px; font-weight: bold; color: #1E293B;">
                            {len(self.df)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    if not self.df.empty and 'DateTime_ET' in self.df.columns:
                        latest_time = self.df['DateTime_ET'].iloc[0].strftime("%H:%M ET")
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #F8FAFC, #F1F5F9); 
                                    padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 14px; color: #64748B; margin-bottom: 5px;">
                                🕒 Latest Update
                            </div>
                            <div style="font-size: 28px; font-weight: bold; color: #1E293B;">
                                {latest_time}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_c:
                    sentiment_color = overall_sentiment['color']
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #F8FAFC, #F1F5F9); 
                                padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 14px; color: #64748B; margin-bottom: 5px;">
                            📈 Overall Sentiment
                        </div>
                        <div style="font-size: 28px; font-weight: bold; color: {sentiment_color};">
                            {overall_sentiment['score']:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.info("📭 No news items to display.")
