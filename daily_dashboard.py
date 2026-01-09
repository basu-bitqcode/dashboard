import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ===================================================================
# 🧮 Helper Functions (Daily specific)
# ===================================================================
def format_currency(val, currency_symbol="$"):
    if pd.isna(val) or val == 0:
        return f"{currency_symbol}0.00"
    return f"{currency_symbol}{val:,.2f}"

def format_inr(val):
    """Format Indian Rupees"""
    if pd.isna(val) or val == 0:
        return "₹0.00"
    return f"₹{val:,.2f}"

def get_currency_formatter(region):
    """Return appropriate currency formatter based on region"""
    return lambda x: format_inr(x) if region == "INDIA" else lambda x: format_currency(x, "$")

def get_currency_symbol(region):
    """Return currency symbol based on region"""
    return "₹" if region == "INDIA" else "$"

def create_metric_card(title, value, value_color="#000000"):
    """Create a metric card with consistent styling"""
    return f"""
    <div style="text-align: center;">
        <div style="font-size: 0.85rem; font-weight: 600; color: {value_color}; margin-bottom: 0.2rem;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {value_color};">{value}</div>
    </div>
    """

def create_daily_pnl_chart(daily_pnl_df, currency_symbol):
    """Create IMPROVED daily P&L chart with smooth line and markers - MATCHING INTRA STYLE"""
    if daily_pnl_df.empty:
        return None
    
    # Sort by date
    daily_pnl_df_sorted = daily_pnl_df.sort_values('Date', ascending=True).copy()
    
    # Ensure we have required columns
    required_cols = ['Net P&L', 'Cumulative Net P&L']
    for col in required_cols:
        if col not in daily_pnl_df_sorted.columns:
            st.warning(f"Missing required column: {col}")
            return None
    
    # Create date strings for display
    daily_pnl_df_sorted['Date_Str'] = daily_pnl_df_sorted['Date'].dt.strftime('%Y-%m-%d')
    
    # Calculate key metrics
    highest_value = daily_pnl_df_sorted['Cumulative Net P&L'].max()
    lowest_value = daily_pnl_df_sorted['Cumulative Net P&L'].min()
    latest_value = daily_pnl_df_sorted['Cumulative Net P&L'].iloc[-1]
    
    # Find indices of extremes
    highest_idx = daily_pnl_df_sorted['Cumulative Net P&L'].idxmax()
    lowest_idx = daily_pnl_df_sorted['Cumulative Net P&L'].idxmin()
    
    fig = go.Figure()
    
    # Create color segments for cumulative line (like intraday chart)
    segments = []
    current_segment = {'x': [], 'y': [], 'color': None}
    
    for i in range(len(daily_pnl_df_sorted)):
        current_val = daily_pnl_df_sorted['Cumulative Net P&L'].iloc[i]
        current_date = daily_pnl_df_sorted['Date_Str'].iloc[i]
        current_color = '#10B981' if current_val >= 0 else '#EF4444'
        
        if not current_segment['x']:
            current_segment['x'].append(current_date)
            current_segment['y'].append(current_val)
            current_segment['color'] = current_color
        elif current_segment['color'] == current_color:
            current_segment['x'].append(current_date)
            current_segment['y'].append(current_val)
        else:
            segments.append(current_segment.copy())
            current_segment = {
                'x': [current_date],
                'y': [current_val],
                'color': current_color
            }
    
    if current_segment['x']:
        segments.append(current_segment)
    
    # Add colored segments
    for segment in segments:
        fig.add_trace(go.Scatter(
            x=segment['x'],
            y=segment['y'],
            mode='lines',
            line=dict(shape='spline', smoothing=0.8, width=4, color=segment['color']),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add invisible trace for hover on cumulative line
    fig.add_trace(go.Scatter(
        x=daily_pnl_df_sorted['Date_Str'],
        y=daily_pnl_df_sorted['Cumulative Net P&L'],
        mode='lines',
        line=dict(width=0),
        hovertemplate=f'<b>%{{x}}</b><br>Cumulative P&L: {currency_symbol}%{{y:,.2f}}<extra></extra>',
        showlegend=False,
        name='Cumulative P&L'
    ))
    
    # Add daily bars with gradient colors based on value
    daily_colors = []
    for val in daily_pnl_df_sorted['Net P&L']:
        if val >= 0:
            # Green gradient: darker for higher values
            intensity = min(0.7 + (val / daily_pnl_df_sorted['Net P&L'].abs().max() * 0.3), 1.0)
            daily_colors.append(f'rgba(16, 185, 129, {intensity})')
        else:
            # Red gradient: darker for lower values
            intensity = min(0.7 + (abs(val) / daily_pnl_df_sorted['Net P&L'].abs().max() * 0.3), 1.0)
            daily_colors.append(f'rgba(239, 68, 68, {intensity})')
    
    fig.add_trace(go.Bar(
        x=daily_pnl_df_sorted['Date_Str'],
        y=daily_pnl_df_sorted['Net P&L'],
        name='Daily P&L',
        marker_color=daily_colors,
        opacity=0.8,
        hovertemplate=f'<b>%{{x}}</b><br>Daily P&L: {currency_symbol}%{{y:,.2f}}<extra></extra>',
        yaxis='y2'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1, opacity=0.3)
    
    # Add area fill for cumulative line
    x_full = daily_pnl_df_sorted['Date_Str'].tolist()
    y_full = daily_pnl_df_sorted['Cumulative Net P&L'].tolist()
    
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
    
    # Add extreme points for cumulative line
    if not pd.isna(highest_idx):
        highest_date = daily_pnl_df_sorted.loc[highest_idx, 'Date_Str']
        fig.add_trace(go.Scatter(
            x=[highest_date],
            y=[highest_value],
            mode='markers+text',
            marker=dict(size=14, color='#10B981', symbol='triangle-up', line=dict(width=2, color='white')),
            text=[f"  High: {currency_symbol}{highest_value:,.0f}"],
            textposition="top center",
            textfont=dict(size=12, color='#10B981', family='Arial'),
            hovertemplate=f'<b>Highest: {currency_symbol}{highest_value:,.2f}</b><br>Date: %{{x}}<extra></extra>',
            showlegend=False,
            name='Highest'
        ))
    
    if not pd.isna(lowest_idx):
        lowest_date = daily_pnl_df_sorted.loc[lowest_idx, 'Date_Str']
        fig.add_trace(go.Scatter(
            x=[lowest_date],
            y=[lowest_value],
            mode='markers+text',
            marker=dict(size=14, color='#EF4444', symbol='triangle-down', line=dict(width=2, color='white')),
            text=[f"  Low: {currency_symbol}{lowest_value:,.0f}"],
            textposition="bottom center",
            textfont=dict(size=12, color='#EF4444', family='Arial'),
            hovertemplate=f'<b>Lowest: {currency_symbol}{lowest_value:,.2f}</b><br>Date: %{{x}}<extra></extra>',
            showlegend=False,
            name='Lowest'
        ))
    
    # Add current marker
    latest_date = daily_pnl_df_sorted['Date_Str'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[latest_date],
        y=[latest_value],
        mode='markers+text',
        marker=dict(size=12, color='#3B82F6', symbol='star', line=dict(width=2, color='white')),
        text=[f"  Current: {currency_symbol}{latest_value:,.0f}"],
        textposition="top right",
        textfont=dict(size=12, color='#3B82F6', family='Arial'),
        hovertemplate=f'<b>Current: {currency_symbol}{latest_value:,.2f}</b><br>Date: %{{x}}<extra></extra>',
        showlegend=False,
        name='Current'
    ))
    
    # Update layout
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=40, b=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        bargap=0.15,
        xaxis=dict(
            title="Date",
            type='category',
            categoryorder='category ascending',
            showgrid=False,
            tickfont=dict(size=11, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True,
            tickangle=45 if len(daily_pnl_df_sorted) > 5 else 0,
            tickmode='array',
            tickvals=daily_pnl_df_sorted['Date_Str'].tolist(),
            ticktext=daily_pnl_df_sorted['Date_Str'].tolist()
        ),
        yaxis=dict(
            title=f"Cumulative P&L ({currency_symbol})",
            side="left",
            showgrid=True,
            gridcolor='#F1F5F9',
            gridwidth=1,
            tickprefix=currency_symbol,
            tickformat=',.0f',
            tickfont=dict(size=11, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True,
            zeroline=False
        ),
        yaxis2=dict(
            title=f"Daily P&L ({currency_symbol})",
            side="right",
            overlaying="y",
            showgrid=False,
            tickprefix=currency_symbol,
            tickformat=',.0f',
            tickfont=dict(size=11, color='#64748B'),
            linecolor='#E2E8F0',
            showline=True,
            zeroline=False
        )
    )
    
    return fig

def create_daily_pnl_dashboard(daily_pnl_df, region="INDIA"):
    """Create Daily PnL dashboard with improved visualizations"""
    if daily_pnl_df.empty:
        st.info(f"📭 No Daily PnL data available for {region}")
        return
    
    format_currency_func = get_currency_formatter(region)
    currency_symbol = get_currency_symbol(region)
    
    # Sort data
    daily_pnl_sorted = daily_pnl_df.sort_values('Date', ascending=True)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_gross = daily_pnl_df['Gross P&L'].sum()
        gross_color = "#2ca02c" if total_gross >= 0 else "#d62728"
        st.markdown(create_metric_card("Total Gross P&L", format_currency_func(total_gross), gross_color), unsafe_allow_html=True)
    
    with col2:
        total_charges = daily_pnl_df['Charges'].sum()
        st.markdown(create_metric_card("Total Charges", format_currency_func(total_charges), "#d62728"), unsafe_allow_html=True)
    
    with col3:
        total_net = daily_pnl_df['Net P&L'].sum()
        net_color = "#10B981" if total_net >= 0 else "#EF4444"
        st.markdown(create_metric_card("Total Net P&L", format_currency_func(total_net), net_color), unsafe_allow_html=True)
    
    with col4:
        if not daily_pnl_sorted.empty:
            current_cumulative = daily_pnl_sorted['Cumulative Net P&L'].iloc[-1]
            cumulative_color = "#10B981" if current_cumulative >= 0 else "#EF4444"
            st.markdown(create_metric_card("Cumulative P&L", format_currency_func(current_cumulative), cumulative_color), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Cumulative P&L", "N/A", "#FFA500"), unsafe_allow_html=True)
    
    st.divider()
    
    # Display daily P&L chart
    fig = create_daily_pnl_chart(daily_pnl_df, currency_symbol)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics section
    st.divider()
    st.subheader("📈 Performance Statistics")
    
    # Calculate statistics
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        winning_days = len(daily_pnl_df[daily_pnl_df['Net P&L'] > 0])
        total_days = len(daily_pnl_df)
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with stats_col2:
        avg_win = daily_pnl_df[daily_pnl_df['Net P&L'] > 0]['Net P&L'].mean()
        avg_win_display = format_currency_func(avg_win) if not pd.isna(avg_win) else "N/A"
        st.metric("Avg Win", avg_win_display)
    
    with stats_col3:
        avg_loss = daily_pnl_df[daily_pnl_df['Net P&L'] < 0]['Net P&L'].mean()
        avg_loss_display = format_currency_func(avg_loss) if not pd.isna(avg_loss) else "N/A"
        st.metric("Avg Loss", avg_loss_display)
    
    with stats_col4:
        if not pd.isna(avg_win) and not pd.isna(avg_loss) and avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        else:
            st.metric("Profit Factor", "N/A")
    
    # Display data table
    st.divider()
    st.subheader("📋 Daily Performance Details")
    
    display_df = daily_pnl_df.copy()
    display_df = display_df.sort_values('Date', ascending=False)
    
    # Format columns for display
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    for col in ['Gross P&L', 'Charges', 'Net P&L', 'Cumulative Net P&L']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_currency_func)
    
    # Show only recent data
    st.dataframe(
        display_df.head(30),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Gross P&L": st.column_config.TextColumn("Gross P&L", width="medium"),
            "Charges": st.column_config.TextColumn("Charges", width="medium"),
            "Net P&L": st.column_config.TextColumn("Net P&L", width="medium"),
            "Cumulative Net P&L": st.column_config.TextColumn("Cumulative", width="medium")
        }
    )
