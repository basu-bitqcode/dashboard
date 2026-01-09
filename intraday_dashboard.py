import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import pytz

# ===================================================================
# 🧮 Helper Functions (Intraday specific)
# ===================================================================
def get_time_with_timezone(region):
    """Get current time with appropriate timezone"""
    if region == "INDIA":
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S IST')
    else:
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S ET')

def get_currency_formatter(region):
    """Return appropriate currency formatter based on region"""
    return lambda x: format_inr(x) if region == "INDIA" else lambda x: format_currency(x, "$")

def get_currency_symbol(region):
    """Return currency symbol based on region"""
    return "₹" if region == "INDIA" else "$"

def format_currency(val, currency_symbol="$"):
    if pd.isna(val) or val == 0:
        return f"{currency_symbol}0.00"
    return f"{currency_symbol}{val:,.2f}"

def format_inr(val):
    """Format Indian Rupees"""
    if pd.isna(val) or val == 0:
        return "₹0.00"
    return f"₹{val:,.2f}"

def create_metric_card(title, value, value_color="#000000"):
    """Create a metric card with consistent styling"""
    return f"""
    <div style="text-align: center;">
        <div style="font-size: 0.85rem; font-weight: 600; color: {value_color}; margin-bottom: 0.2rem;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {value_color};">{value}</div>
    </div>
    """

def get_pnl_color(value, currency_symbol):
    """Return color for P&L values"""
    if pd.isna(value) or value == 0:
        return "gray"
    str_value = str(value)
    if f"{currency_symbol}-" in str_value or "-" in str_value or "−" in str_value:
        return "red"
    return "green"

def create_html_table(df, columns, currency_symbol):
    """Create HTML table from dataframe with consistent styling"""
    if df.empty:
        return ""
    
    # Create header
    html = """
    <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <thead>
            <tr style="background-color: #f2f2f2;">
    """
    
    for col in columns:
        align = "right" if col in ['Quantity', 'Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure', 
                                  'Buy Qty', 'Buy Price', 'Sell Qty', 'Sell Price', 'Realized P&L'] else "left"
        html += f'<th style="padding: 10px; text-align: {align}; border-bottom: 1px solid #ddd;">{col}</th>'
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Add rows
    for _, row in df.iterrows():
        html += '<tr>'
        for col in columns:
            align = "right" if col in ['Quantity', 'Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure',
                                      'Buy Qty', 'Buy Price', 'Sell Qty', 'Sell Price', 'Realized P&L'] else "left"
            
            cell_value = row[col]
            cell_style = f"padding: 8px; border-bottom: 1px solid #ddd; text-align: {align};"
            
            # Apply color coding for P&L columns
            if col in ['Unrealized P&L', 'Realized P&L']:
                pnl_color = get_pnl_color(cell_value, currency_symbol)
                cell_style += f" font-weight: bold; color: {pnl_color};"
            
            html += f'<td style="{cell_style}">{cell_value}</td>'
        
        html += '</tr>'
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html

def create_live_pnl_chart(live_pnl_df, currency_symbol):
    """Create live P&L chart with color transitions - SMOOTH VERSION"""
    if live_pnl_df.empty:
        return None
    
    live_pnl_df_sorted = live_pnl_df.sort_values('DateTime')
    highest_value = live_pnl_df_sorted['Total PnL'].max()
    lowest_value = live_pnl_df_sorted['Total PnL'].min()
    highest_row = live_pnl_df_sorted[live_pnl_df_sorted['Total PnL'] == highest_value].iloc[0]
    lowest_row = live_pnl_df_sorted[live_pnl_df_sorted['Total PnL'] == lowest_value].iloc[0]
    
    fig = go.Figure()
    segments = []
    current_segment = {'x': [], 'y': [], 'color': None}
    
    for i in range(len(live_pnl_df_sorted)):
        current_val = live_pnl_df_sorted['Total PnL'].iloc[i]
        current_time = live_pnl_df_sorted['DateTime'].iloc[i]
        current_color = '#10B981' if current_val >= 0 else '#EF4444'
        
        if not current_segment['x']:
            current_segment['x'].append(current_time)
            current_segment['y'].append(current_val)
            current_segment['color'] = current_color
        elif current_segment['color'] == current_color:
            current_segment['x'].append(current_time)
            current_segment['y'].append(current_val)
        else:
            prev_val = live_pnl_df_sorted['Total PnL'].iloc[i-1]
            prev_time = live_pnl_df_sorted['DateTime'].iloc[i-1]
            m = (current_val - prev_val) / ((current_time - prev_time).total_seconds())
            zero_time_seconds = -prev_val / m if m != 0 else 0
            zero_time = prev_time + pd.Timedelta(seconds=zero_time_seconds)
            
            current_segment['x'].append(zero_time)
            current_segment['y'].append(0)
            segments.append(current_segment.copy())
            current_segment = {
                'x': [zero_time, current_time],
                'y': [0, current_val],
                'color': current_color
            }
    
    if current_segment['x']:
        segments.append(current_segment)
    
    for segment in segments:
        fig.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(shape='spline', smoothing=1.0, width=3, color=segment['color']),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add invisible trace for hover
    fig.add_trace(go.Scatter(
        x=live_pnl_df_sorted['DateTime'],
        y=live_pnl_df_sorted['Total PnL'],
        mode='lines',
        line=dict(width=0),
        hovertemplate=f'<b>%{{x|%H:%M:%S}}</b><br>{currency_symbol}%{{y:,.2f}}<extra></extra>',
        showlegend=False,
        name='Live P&L'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1, opacity=0.3)
    
    # Add area fill
    x_full = live_pnl_df_sorted['DateTime'].tolist()
    y_full = live_pnl_df_sorted['Total PnL'].tolist()
    
    fig.add_trace(go.Scatter(
        x=x_full,
        y=y_full,
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_full,
        y=[min(y, 0) for y in y_full],
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add extreme points
    fig.add_trace(go.Scatter(
        x=[highest_row['DateTime']],
        y=[highest_value],
        mode='markers+text',
        marker=dict(size=12, color='#10B981', symbol='triangle-up', line=dict(width=2, color='white')),
        text=[f"  High: {currency_symbol}{highest_value:,.0f}"],
        textposition="top center",
        textfont=dict(size=11, color='#10B981', family='Arial'),
        hovertemplate=f'<b>Highest: {currency_symbol}{highest_value:,.2f}</b><br>Time: %{{x|%H:%M:%S}}<extra></extra>',
        showlegend=False,
        name='Highest'
    ))
    
    fig.add_trace(go.Scatter(
        x=[lowest_row['DateTime']],
        y=[lowest_value],
        mode='markers+text',
        marker=dict(size=12, color='#EF4444', symbol='triangle-down', line=dict(width=2, color='white')),
        text=[f"  Low: {currency_symbol}{lowest_value:,.0f}"],
        textposition="bottom center",
        textfont=dict(size=11, color='#EF4444', family='Arial'),
        hovertemplate=f'<b>Lowest: {currency_symbol}{lowest_value:,.2f}</b><br>Time: %{{x|%H:%M:%S}}<extra></extra>',
        showlegend=False,
        name='Lowest'
    ))
    
    # Update layout
    fig.update_layout(
        height=380,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=20, b=40),
        xaxis=dict(
            showgrid=False,
            tickformat='%H:%M',
            tickfont=dict(size=10, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#F1F5F9',
            gridwidth=1,
            tickprefix=currency_symbol,
            tickformat=',.0f',
            tickfont=dict(size=10, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True
        ),
        showlegend=False
    )
    
    return fig

def create_intraday_dashboard(data_dict, live_pnl_df, region="INDIA"):
    """Create intraday dashboard for either INDIA or GLOBAL region"""
    open_df = data_dict['open_positions']
    closed_df = data_dict['closed_positions']
    summary = data_dict['summary']
    
    if open_df.empty and closed_df.empty:
        st.info(f"📭 NO ACTIVE POSITIONS.")
        return
    
    format_currency_func = get_currency_formatter(region)
    currency_symbol = get_currency_symbol(region)
    
    # Display total P&L
    total_pnl = summary.get('total_pnl', 0)
    pnl_color = "green" if total_pnl > 0 else "red" if total_pnl < 0 else "gray"
    
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <span style="font-size: 2.4rem; font-weight: 800; color: {pnl_color};">
                {format_currency_func(total_pnl)}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(create_metric_card("Closed P&L", format_currency_func(summary.get('total_closed_pnl', 0)), "#1f77b4"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Unrealized P&L", format_currency_func(summary.get('total_unrealized_pnl', 0)), "#ff7f0e"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Traded Volume", format_currency_func(summary.get('total_traded_volume', 0)), "#2ca02c"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Open Exposure", format_currency_func(summary.get('total_open_exposure', 0)), "#d62728"), unsafe_allow_html=True)
    with col5:
        st.markdown(create_metric_card("Open Positions", summary.get('open_positions_count', 0), "#9467bd"), unsafe_allow_html=True)
    with col6:
        st.markdown(create_metric_card("Closed Positions", summary.get('closed_positions_count', 0), "#8c564b"), unsafe_allow_html=True)
    
    # Show last updated time
    if not live_pnl_df.empty and 'DateTime' in live_pnl_df.columns:
        last_datetime = live_pnl_df['DateTime'].iloc[-1]
        timezone_str = "IST" if region == "INDIA" else "ET"
        formatted_time = last_datetime.strftime(f'%Y-%m-%d %H:%M:%S {timezone_str}')
        st.caption(f"📊 Last Updated: {formatted_time}")
    
    # Display live P&L chart
    if not live_pnl_df.empty:
        st.divider()
        fig = create_live_pnl_chart(live_pnl_df, currency_symbol)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Display open positions
    if not open_df.empty:
        st.divider()
        st.subheader("📈 Open Positions")
        
        open_display_df = open_df[[
            'tradingsymbol', 'position_type', 'net_quantity',
            'avg_price', 'last_price', 'unrealized_pnl', 'open_exposure'
        ]].copy()
        
        open_display_df = open_display_df.rename(columns={
            'tradingsymbol': 'Symbol',
            'position_type': 'Position',
            'net_quantity': 'Quantity',
            'avg_price': 'Avg Price',
            'last_price': 'Last Price',
            'unrealized_pnl': 'Unrealized P&L',
            'open_exposure': 'Open Exposure'
        })
        
        # Format columns
        for col in ['Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure']:
            open_display_df[col] = open_display_df[col].apply(format_currency_func)
        
        table_html = create_html_table(
            open_display_df,
            ['Symbol', 'Position', 'Quantity', 'Avg Price', 'Last Price', 'Unrealized P&L', 'Open Exposure'],
            currency_symbol
        )
        st.markdown(table_html, unsafe_allow_html=True)
    
    # Display closed positions
    if not closed_df.empty:
        st.divider()
        st.subheader("📊 Closed Positions (Today)")
        
        # Sort by P&L BEFORE any processing
        closed_df_sorted = closed_df.sort_values(by='pnl', ascending=False)
        
        closed_display_df = closed_df_sorted[[
            'tradingsymbol', 'buy_quantity', 'buy_price',
            'sell_quantity', 'sell_price', 'pnl'
        ]].copy()
        
        closed_display_df = closed_display_df.rename(columns={
            'tradingsymbol': 'Symbol',
            'buy_quantity': 'Buy Qty',
            'buy_price': 'Buy Price',
            'sell_quantity': 'Sell Qty',
            'sell_price': 'Sell Price',
            'pnl': 'Realized P&L'
        })
        
        # Format columns AFTER sorting
        for col in ['Buy Price', 'Sell Price', 'Realized P&L']:
            closed_display_df[col] = closed_display_df[col].apply(format_currency_func)
        
        table_html = create_html_table(
            closed_display_df,
            ['Symbol', 'Buy Qty', 'Buy Price', 'Sell Qty', 'Sell Price', 'Realized P&L'],
            currency_symbol
        )
        st.markdown(table_html, unsafe_allow_html=True)
