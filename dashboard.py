import streamlit as st
import pandas as pd
import numpy as np
from news_sentiment_module import NewsSentimentAnalyzer
from intraday_dashboard import create_intraday_dashboard
from daily_dashboard import create_daily_pnl_dashboard

# ===================================================================
# 🛠️ CONFIGURATION
# ===================================================================
REFRESH_INTERVAL_SEC = 10
CACHE_TTL = 10

# ===================================================================
# 🔐 Load Google Sheet URL from Streamlit Secrets
# ===================================================================
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
    NEWS_SHEET_URL = st.secrets.get("news_sheet", {}).get("url", "")

except KeyError:
    st.error("🔐 Missing Google Sheet URL in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="BITQCODE Dashboard",
    page_icon="💼",
    layout="wide"
)

# ===================================================================
# 📥 Data Loading & Processing (Keep only data loading functions)
# ===================================================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC, show_spinner=False)
def load_sheet_data(sheet_gid="0"):
    """Load specific sheet from Google Sheets using gid parameter"""
    try:
        if "export?format=csv" in GOOGLE_SHEET_CSV_URL:
            if "gid=" in GOOGLE_SHEET_CSV_URL:
                url = GOOGLE_SHEET_CSV_URL.split("&gid=")[0] + f"&gid={sheet_gid}"
            else:
                url = GOOGLE_SHEET_CSV_URL + f"&gid={sheet_gid}"
        else:
            url = GOOGLE_SHEET_CSV_URL + f"?gid={sheet_gid}&format=csv"
        
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"❌ Failed to load sheet {sheet_gid}: {str(e)[:150]}...")
        return pd.DataFrame()

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_live_pnl_data(df_raw):
    """Process Live PnL data - filter for latest date only"""
    if df_raw.empty:
        return pd.DataFrame()
    
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    required_cols = ['DateTime', 'Total PnL']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['Total PnL'] = pd.to_numeric(df['Total PnL'], errors='coerce')
    df = df.dropna(subset=['DateTime', 'Total PnL'])
    
    if df.empty:
        return df
    
    df['Date'] = df['DateTime'].dt.date
    latest_date = df['Date'].max()
    df_today = df[df['Date'] == latest_date].copy()
    df_today = df_today.sort_values('DateTime')
    
    return df_today

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_india_data(df_raw):
    """Process INDIA data with the new format"""
    if df_raw.empty:
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    expected_cols = [
        's_no', 'tradingsymbol', 'buy_value', 'buy_price', 
        'buy_quantity', 'sell_quantity', 'sell_price', 
        'sell_value', 'last_price', 'pnl'
    ]
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    numeric_cols = ['buy_value', 'buy_price', 'buy_quantity', 'sell_quantity', 
                   'sell_price', 'sell_value', 'last_price', 'pnl']
    
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.strip()
            .replace('', '0')
            .astype(float)
        )
    
    df = df.dropna(subset=['tradingsymbol'])
    df = df[df['tradingsymbol'].astype(str).str.strip() != '']
    
    # Separate open and closed positions
    closed_mask = (df['buy_quantity'] > 0) & (df['sell_quantity'] > 0) & (df['buy_quantity'] == df['sell_quantity'])
    closed_df = df[closed_mask].copy()
    
    closed_df['pnl'] = (closed_df['sell_price'] - closed_df['buy_price']) * closed_df['sell_quantity']
    
    open_mask = ~closed_mask
    open_df = df[open_mask].copy()
    
    # Calculate additional metrics for open positions
    if not open_df.empty:
        open_df['net_quantity'] = open_df['buy_quantity'] - open_df['sell_quantity']
        
        open_df['avg_price'] = np.where(
            open_df['net_quantity'] != 0,
            (open_df['buy_value'] - open_df['sell_value']) / open_df['net_quantity'],
            0
        )
        
        open_df['unrealized_pnl'] = (open_df['last_price'] - open_df['avg_price']) * open_df['net_quantity']
        open_df['open_exposure'] = open_df['net_quantity'] * open_df['last_price']
        open_df['position_type'] = open_df['net_quantity'].apply(lambda x: 'Long' if x > 0 else 'Short' if x < 0 else 'Flat')
        open_df = open_df.sort_values('unrealized_pnl', ascending=False)
    
    # Calculate summary metrics
    total_traded_volume = df['buy_value'].sum() + df['sell_value'].sum()
    total_closed_pnl = closed_df['pnl'].sum() if not closed_df.empty else 0
    total_unrealized_pnl = open_df['unrealized_pnl'].sum() if not open_df.empty else 0
    total_open_exposure = open_df['open_exposure'].abs().sum() if not open_df.empty else 0
    
    return {
        'open_positions': open_df,
        'closed_positions': closed_df,
        'summary': {
            'total_traded_volume': total_traded_volume,
            'total_closed_pnl': total_closed_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_open_exposure': total_open_exposure,
            'open_positions_count': len(open_df),
            'closed_positions_count': len(closed_df),
            'total_pnl': total_closed_pnl + total_unrealized_pnl
        }
    }

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_daily_pnl_data(df_raw, region="INDIA"):
    """Process Daily PnL data for INDIA and GLOBAL"""
    if df_raw.empty:
        return pd.DataFrame()
    
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    required_cols = ['Date', 'Gross P&L', 'Charges', 'Net P&L']
    
    # Try alternative column names
    alternative_cols = {
        'Gross P&L': ['Gross PnL', 'GrossPnL', 'Gross'],
        'Charges': ['Fees', 'Commission', 'Brokerage'],
        'Net P&L': ['Net PnL', 'NetPnL', 'Net']
    }
    
    for required, alternatives in alternative_cols.items():
        if required not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df[required] = df[alt]
                    break
    
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    for col in ['Gross P&L', 'Charges', 'Net P&L']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('Date', ascending=True)
    df['Cumulative Gross P&L'] = df['Gross P&L'].cumsum()
    df['Cumulative Net P&L'] = df['Net P&L'].cumsum()
    df['Date_Display'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Region'] = region
    
    return df

def create_refresh_button(key_suffix):
    """Create a refresh button that clears cache"""
    if st.button("🔄 Refresh Data", type="secondary", key=f"refresh_{key_suffix}"):
        st.cache_data.clear()
        st.rerun()

# ===================================================================
# 🎨 CSS for Bigger Tabs
# ===================================================================
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        padding-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        font-size: 16px;
        font-weight: 600;
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
        background-color: #f0f2f6;
        white-space: nowrap;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #FF4B4B;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #e0e0e0;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    .main .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# 📥 Load Data
# ===================================================================
df_india_raw = load_sheet_data(sheet_gid="649765105")
df_india_live_pnl_raw = load_sheet_data(sheet_gid="1065660372")
df_india_daily_pnl_raw = load_sheet_data(sheet_gid="795838620")

df_global_raw = load_sheet_data(sheet_gid="94252270")
df_global_live_pnl_raw = load_sheet_data(sheet_gid="1297846329")
df_global_daily_pnl_raw = load_sheet_data(sheet_gid="563240267")

# Process data
india_data = process_india_data(df_india_raw)
india_live_pnl_data = process_live_pnl_data(df_india_live_pnl_raw)
india_daily_pnl_data = process_daily_pnl_data(df_india_daily_pnl_raw, region="INDIA")

global_data = process_india_data(df_global_raw) if not df_global_raw.empty else {
    'open_positions': pd.DataFrame(), 'closed_positions': pd.DataFrame(), 'summary': {}
}
global_live_pnl_data = process_live_pnl_data(df_global_live_pnl_raw)
global_daily_pnl_data = process_daily_pnl_data(df_global_daily_pnl_raw, region="GLOBAL")

# ===================================================================
# 📊 Create Tabs
# ===================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 **GLOBAL (INTRA)**", 
    "📊 **GLOBAL (DAILY)**",
    "🇮🇳 **INDIA (INTRA)**",
    "📊 **INDIA (DAILY)**",
    "📰 **NEWS & SENTIMENT**"
])

with tab1:
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("global_intra")
    
    create_intraday_dashboard(global_data, global_live_pnl_data, region="GLOBAL")

with tab2:
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("global_daily")
    
    create_daily_pnl_dashboard(global_daily_pnl_data, region="GLOBAL")

with tab3:
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("india_intra")
    
    create_intraday_dashboard(india_data, india_live_pnl_data, region="INDIA")

with tab4:
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("india_daily")
    
    create_daily_pnl_dashboard(india_daily_pnl_data, region="INDIA")

with tab5:
    # NEWS & SENTIMENT TAB
    col1, col2 = st.columns([5, 1])
    with col2:
        create_refresh_button("news_sentiment")
    
    if not NEWS_SHEET_URL:
        st.warning("""
        ⚠️ News not Updated.
        """)
    else:
        analyzer = NewsSentimentAnalyzer(google_sheet_url=NEWS_SHEET_URL)
        analyzer.display_dashboard()
