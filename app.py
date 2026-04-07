import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Quant Portfolio Dashboard", layout="wide")

# --- 1. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    file_path = "/Users/rhadeshluchmun/Desktop/AFRAE-AM/Project Data.xlsx"
    
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    
    mapping_sheet = [s for s in sheets if "Mapping" in s][0]
    weights_sheet = [s for s in sheets if "Weight" in s][0]
    returns_sheet = [s for s in sheets if "Return" in s][0]
    alphas_sheet = [s for s in sheets if "Alpha" in s][0]

    # Read each sheet using the dynamic names
    mapping = pd.read_excel(file_path, sheet_name=mapping_sheet)
    
    weights = pd.read_excel(file_path, sheet_name=weights_sheet)
    weights.rename(columns={weights.columns[0]: 'Date'}, inplace=True)
    weights['Date'] = pd.to_datetime(weights['Date'])
    
    returns = pd.read_excel(file_path, sheet_name=returns_sheet)
    returns.rename(columns={returns.columns[0]: 'Date'}, inplace=True)
    returns['Date'] = pd.to_datetime(returns['Date'])
    
    alphas = pd.read_excel(file_path, sheet_name=alphas_sheet)
    alphas.rename(columns={alphas.columns[0]: 'Date'}, inplace=True)
    alphas['Date'] = pd.to_datetime(alphas['Date'])

    # Reshape Wide to Long matrices
    weights_long = weights.melt(id_vars=['Date'], var_name='Country', value_name='Weight')
    returns_long = returns.melt(id_vars=['Date'], var_name='Ticker', value_name='Return')
    alphas_long = alphas.melt(id_vars=['Date'], var_name='Ticker', value_name='Alpha')

    # Merge Mapping onto returns and alphas
    returns_long = returns_long.merge(mapping, on='Ticker', how='left')
    alphas_long = alphas_long.merge(mapping, on='Ticker', how='left')

    return mapping, weights_long, returns_long, alphas_long

mapping, weights_long, returns_long, alphas_long = load_data()

st.title("Quantitative Portfolio & Factor Dashboard")

tab1, tab2, tab3 = st.tabs(["Country Allocation", "Portfolio Analytics", "Alpha Analysis"])

# --- 2. COUNTRY ALLOCATION ---
with tab1:
    st.header("Country Allocation Analysis")
    
    # Split the layout into two columns for better use of wide screen
    col_trend, col_snap = st.columns(2)
    
    with col_trend:
        st.subheader("Historical Weight Trend")
        # Using a standard line chart prevents the "stacking distortion" of area charts
        fig_trend = px.line(weights_long, x='Date', y='Weight', color='Country', 
                            title="Individual Country Weights Over Time")
        fig_trend.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_snap:
        st.subheader("Point-in-Time Snapshot")
        # Allow user to select a specific date to view exact weightings clearly
        available_dates = sorted(weights_long['Date'].dropna().unique())
        selected_date = st.selectbox("Select Date to View Exact Weights", available_dates, index=len(available_dates)-1)
        
        # Filter to the selected date and sort highest to lowest weight
        snapshot_data = weights_long[weights_long['Date'] == selected_date].sort_values(by='Weight', ascending=False)
        
        # A simple bar chart is the easiest way to interpret static comparisons
        fig_bar = px.bar(snapshot_data, x='Country', y='Weight', color='Country',
                         title=f"Country Weights on {selected_date.strftime('%Y-%m-%d')}")
        fig_bar.update_traces(texttemplate='%{y:.2%}', textposition='outside')
        fig_bar.update_layout(yaxis_tickformat='.1%', showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- 3. PORTFOLIO CONSTRUCTION & ANALYTICS ---
with tab2:
    st.header("Portfolio Performance & Risk")
    
    # Calculate Equal-Weighted Country Returns
    ew_country_returns = returns_long.groupby(['Date', 'Country'])['Return'].mean().reset_index()
    
    # Merge with weights to calculate overall portfolio return
    port_data = pd.merge(ew_country_returns, weights_long, on=['Date', 'Country'])
    port_data['Weighted_Return'] = port_data['Return'] * port_data['Weight']
    
    # Aggregate to total portfolio level
    total_port = port_data.groupby('Date')['Weighted_Return'].sum().reset_index()
    total_port = total_port.sort_values('Date')
    total_port['Cum_Return'] = (1 + total_port['Weighted_Return']).cumprod() - 1
    total_port['High_Water_Mark'] = (1 + total_port['Cum_Return']).cummax()
    total_port['Drawdown'] = (1 + total_port['Cum_Return']) / total_port['High_Water_Mark'] - 1

    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics calculations
    ytd_return = total_port[total_port['Date'].dt.year == total_port['Date'].dt.year.max()]['Weighted_Return'].add(1).prod() - 1
    var_95 = np.percentile(total_port['Weighted_Return'].dropna(), 5)
    sharpe = (total_port['Weighted_Return'].mean() / total_port['Weighted_Return'].std()) * np.sqrt(12) # Assuming monthly data
    max_dd = total_port['Drawdown'].min()

    col1.metric("YTD Return", f"{ytd_return:.2%}")
    col2.metric("Historical VaR (95%)", f"{var_95:.2%}")
    col3.metric("Sharpe Ratio (Ann.)", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_dd:.2%}")

    # Plot Cumulative Return & Drawdown
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=total_port['Date'], y=total_port['Cum_Return'], mode='lines', name='Cumulative Return'))
    fig_perf.add_trace(go.Scatter(x=total_port['Date'], y=total_port['Drawdown'], mode='lines', fill='tozeroy', name='Drawdown', yaxis='y2', opacity=0.3, line=dict(color='red')))
    
    fig_perf.update_layout(title="Portfolio Growth & Drawdowns", yaxis2=dict(title="Drawdown", overlaying='y', side='right'))
    st.plotly_chart(fig_perf, use_container_width=True)

# --- 4. ALPHA ANALYSIS ---
with tab3:
    st.header("Top Alpha Signal Explorer")
    
    available_dates = sorted(alphas_long['Date'].dropna().unique())
    col_a, col_b = st.columns(2)
    selected_date = col_a.selectbox("Select Signal Date", available_dates, index=len(available_dates)-1)
    
    available_countries = ['All'] + list(mapping['Country'].dropna().unique())
    selected_country = col_b.selectbox("Filter by Country", available_countries)

    # Filter data
    signal_data = alphas_long[alphas_long['Date'] == selected_date].dropna(subset=['Alpha'])
    if selected_country != 'All':
        signal_data = signal_data[signal_data['Country'] == selected_country]

    # Rank and slice
    top_n = st.slider("Select Top N Securities by Alpha", 5, 50, 10)
    top_securities = signal_data.nlargest(top_n, 'Alpha')
    
    st.dataframe(top_securities[['Ticker', 'Country', 'Sector', 'Alpha']].reset_index(drop=True), use_container_width=True)

    # Future Performance Visualization Logic
    st.subheader(f"Forward Returns of Top {top_n} Stocks")
    future_returns = returns_long[(returns_long['Date'] > selected_date) & (returns_long['Ticker'].isin(top_securities['Ticker']))]
    
    if not future_returns.empty:
        future_cum = future_returns.groupby(['Date', 'Ticker'])['Return'].sum().groupby(level=1).cumsum().reset_index()
        fig_alpha = px.line(future_cum, x='Date', y='Return', color='Ticker', title="Cumulative Forward Returns of Selected Alpha Stocks")
        st.plotly_chart(fig_alpha, use_container_width=True)
    else:
        st.info("No forward return data available for this date.")