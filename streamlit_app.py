import streamlit as st
import pandas as pd
import json
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly
import os
from preprocessor import MCC_CODE, TXN


st.set_page_config(
    page_title="Transactions EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data using preprocessor
@st.cache_data
def load_data():
    txn = TXN()
    txn.load_csv('encrpt_id.csv')
    # Parse datetime first, since bind expects it
    txn.data['has_time'] = txn.data['time'].str.contains(':')
    txn.data['datetime'] = None
    mask = txn.data['has_time']
    txn.data.loc[mask, 'datetime'] = pd.to_datetime(txn.data.loc[mask, 'date'], dayfirst=True, errors='coerce')
    mask_no = ~txn.data['has_time']
    txn.data.loc[mask_no, 'datetime'] = pd.to_datetime(txn.data.loc[mask_no, 'date'] + ' ' + txn.data.loc[mask_no, 'time'], dayfirst=True, errors='coerce')
    txn.data['datetime'] = pd.to_datetime(txn.data['datetime'])
    # Set TransactionDate for preprocessor
    txn.data['TransactionDate'] = txn.data['datetime']

    # Rename for preprocessor and ensure MCC codes remain as 4-digit strings
    txn.data['mccCode'] = txn.data['MCC'].astype(str).str.zfill(4)

    mcc = MCC_CODE()
    mcc.load_json('mcc_code.json')
    txn.bind_mcc_categories(mcc)

    # Handle P2P if MCC == 0
    txn.data.loc[txn.data['MCC'] == 0, 'mcc_category'] = 'P2P Transactions'

    # Keep only essential columns - mcc_category and mcc_category_range contain the same info as category, category_by_range
    df = txn.data[['WalletId', 'amt', 'date', 'time', 'MCC', 'payee', 'mcc_category', 'mcc_category_range']].copy()

    # Derive weekday and hour from separate date and time columns
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True, errors='coerce')
    df['date_only'] = df['datetime'].dt.date  # Create clean date only column
    df['weekday'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    df = df.drop('datetime', axis=1)
    return df

# Perform EDA
def perform_eda(df):
    insights = {}

    # Overall metrics (across all selected users)
    insights['total_transactions'] = len(df)
    insights['total_spending'] = df['amt'].sum()
    insights['avg_transaction'] = df['amt'].mean() if len(df) > 0 else 0

    # Per-user metrics
    user_stats = df.groupby('WalletId')['amt'].agg(['sum', 'mean', 'count']).reset_index()
    user_stats.columns = ['WalletId', 'total_spending', 'avg_transaction', 'transaction_count']
    insights['user_comparison'] = user_stats.to_dict('records')

    # Spending patterns
    if len(df) > 0:
        insights['spending_distribution'] = df['amt'].describe().to_dict()

        # Time of day analysis
        def time_of_day(hour):
            if hour < 6:
                return 'Early Morning (1-6 AM)'
            elif hour < 12:
                return 'Morning (6 AM-12 PM)'
            elif hour < 18:
                return 'Afternoon (12-6 PM)'
            elif hour < 22:
                return 'Evening (6-10 PM)'
            else:
                return 'Late Night (10 PM-1 AM)'

        df['time_of_day'] = df['hour'].apply(time_of_day)
        insights['time_spending_overall'] = df.groupby('time_of_day')['amt'].agg(['sum', 'mean', 'count']).to_dict()

        # Per-user time of day
        user_time_analysis = df.groupby(['WalletId', 'time_of_day'])['amt'].agg(['sum', 'mean', 'count']).reset_index()
        insights['time_spending_by_user'] = user_time_analysis.to_dict('records')

        # Weekday vs Weekend
        df['is_weekend'] = df['weekday'].isin([5, 6])
        insights['weekday_vs_weekend_overall'] = df.groupby('is_weekend')['amt'].agg(['sum', 'mean', 'count']).to_dict()

        # Per-user weekday/weekend
        user_weekday_analysis = df.groupby(['WalletId', 'is_weekend'])['amt'].agg(['sum', 'mean', 'count']).reset_index()
        insights['weekday_vs_weekend_by_user'] = user_weekday_analysis.to_dict('records')

        # Category analysis overall
        insights['categories_overall'] = df.groupby('mcc_category')['amt'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False).head(10).to_dict()

        # Per-user category analysis
        user_category_analysis = df.groupby(['WalletId', 'mcc_category'])['amt'].agg(['sum', 'mean', 'count']).reset_index()
        insights['categories_by_user'] = user_category_analysis.to_dict('records')

        # Payee frequency analysis
        payee_freq = df.groupby('payee')['amt'].agg(['sum', 'mean', 'count']).reset_index()
        payee_freq['percentage'] = (payee_freq['count'] / payee_freq['count'].sum() * 100).round(1)
        payee_freq = payee_freq.sort_values('count', ascending=False).head(20)  # Top 20 payees

        # Add category information to payees
        payee_category_map = df.drop_duplicates('payee')[['payee', 'mcc_category', 'mcc_category_range']].set_index('payee')
        payee_freq = payee_freq.merge(payee_category_map, left_on='payee', right_index=True, how='left')

        insights['frequent_payees'] = payee_freq.to_dict('records')
        insights['total_unique_payees'] = df['payee'].nunique()

        # Payee category distribution (frequency-based)
        payee_cat_freq = payee_freq.groupby('mcc_category')['count'].sum().reset_index()
        payee_cat_freq['percentage'] = (payee_cat_freq['count'] / payee_cat_freq['count'].sum() * 100).round(1)
        payee_cat_freq = payee_cat_freq.sort_values('count', ascending=False)
        insights['payee_category_distribution'] = payee_cat_freq.to_dict('records')

    else:
        insights['spending_distribution'] = {}
        insights['time_spending_overall'] = {}
        insights['time_spending_by_user'] = []
        insights['weekday_vs_weekend_overall'] = {}
        insights['weekday_vs_weekend_by_user'] = []
        insights['categories_overall'] = {}
        insights['categories_by_user'] = []

    # Add frequency analysis
    if len(df) > 0:
        # Daily transaction frequency
        daily_freq = df.groupby('date_only')['amt'].count().reset_index()
        daily_freq.columns = ['date', 'transaction_count']
        insights['daily_transaction_frequency'] = daily_freq.to_dict('records')

        # Hourly transaction frequency
        hourly_freq = df.groupby('hour')['amt'].count().reset_index()
        hourly_freq.columns = ['hour', 'transaction_count']
        insights['hourly_transaction_frequency'] = hourly_freq.to_dict('records')

        # Weekday transaction frequency
        weekday_freq = df.groupby('weekday')['amt'].count().reset_index()
        weekday_freq.columns = ['weekday', 'transaction_count']
        insights['weekday_transaction_frequency'] = weekday_freq.to_dict('records')

        # Time Series Analysis
        # Monthly spending patterns
        df_for_ts = df.copy()
        df_for_ts['month'] = pd.to_datetime(df_for_ts['date_only']).dt.to_period('M')
        monthly_spend = df_for_ts.groupby('month')['amt'].sum().reset_index()
        monthly_spend['month'] = monthly_spend['month'].astype(str)
        insights['monthly_spending_trend'] = monthly_spend.to_dict('records')

        # Week-over-week comparison (use week start dates)
        df_for_ts['week_start'] = pd.to_datetime(df_for_ts['date_only']) - pd.to_timedelta(pd.to_datetime(df_for_ts['date_only']).dt.dayofweek, unit='D')
        weekly_spend = df_for_ts.groupby('week_start')['amt'].sum().reset_index()
        weekly_spend['week'] = weekly_spend['week_start'].dt.strftime('%Y-%m-%d')
        insights['weekly_spending_trend'] = weekly_spend[['week', 'amt']].to_dict('records')

        # Moving averages (7-day rolling average)
        daily_spend_ts = df.groupby('date_only')['amt'].sum().reset_index()
        daily_spend_ts = daily_spend_ts.sort_values('date_only')
        if len(daily_spend_ts) >= 7:
            daily_spend_ts['7_day_moving_avg'] = daily_spend_ts['amt'].rolling(window=7).mean()
            insights['daily_moving_average'] = daily_spend_ts[['date_only', '7_day_moving_avg']].to_dict('records')
        else:
            insights['daily_moving_average'] = []

        # Seasonal patterns by month of year
        monthly_seasonal = pd.to_datetime(df['date_only']).dt.month
        monthly_patterns = df.groupby(monthly_seasonal)['amt'].agg(['sum', 'mean', 'count']).reset_index()
        monthly_patterns['month_name'] = monthly_patterns['date_only'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        insights['seasonal_monthly_patterns'] = monthly_patterns.to_dict('records')

        # Spending variance and trends
        if len(daily_spend_ts) > 1:
            insights['spending_variance'] = daily_spend_ts['amt'].var()
            insights['spending_std'] = daily_spend_ts['amt'].std()
            insights['avg_daily_spend'] = daily_spend_ts['amt'].mean()

            # Trend direction (positive if increasing)
            from scipy import stats
            slope, _, _, _, _ = stats.linregress(range(len(daily_spend_ts)), daily_spend_ts['amt'])
            insights['spending_trend_slope'] = slope
        else:
            insights['spending_variance'] = 0
            insights['spending_std'] = 0
            insights['avg_daily_spend'] = 0
            insights['spending_trend_slope'] = 0

    else:
        insights['daily_transaction_frequency'] = []
        insights['hourly_transaction_frequency'] = []
        insights['weekday_transaction_frequency'] = []
        insights['monthly_spending_trend'] = []
        insights['weekly_spending_trend'] = []
        insights['daily_moving_average'] = []
        insights['seasonal_monthly_patterns'] = []
        insights['spending_variance'] = 0
        insights['spending_std'] = 0
        insights['avg_daily_spend'] = 0
        insights['spending_trend_slope'] = 0

    return insights, df

def create_plots(df, selected_wallets):
    figs = {}
    if len(df) == 0:
        st.warning("No data for selected user.")
        return figs

    # Get unique wallet IDs for color coding
    wallet_color_map = {wallet: f'hsl({i*360/len(selected_wallets)}, 70%, 50%)' for i, wallet in enumerate(selected_wallets)}

    if len(selected_wallets) == 1:
        # Single user plots (existing logic)
        # Category spending bar chart
        cat_sum = df.groupby('mcc_category')['amt'].sum().reset_index()
        fig_cat = px.bar(cat_sum, x='mcc_category', y='amt', title=f'Spending by Category - {selected_wallets[0]}',
                        labels={'amt': 'Amount (INR)', 'mcc_category': 'Category'})
        figs['category'] = fig_cat

        # Daily spending line chart
        try:
            df_temp = df.copy()
            df_temp['datetime'] = pd.to_datetime(df_temp['date'] + ' ' + df_temp['time'], dayfirst=True, errors='coerce')
            daily_sum = df_temp.groupby(df_temp['datetime'].dt.date)['amt'].sum().reset_index()
            daily_sum['date'] = pd.to_datetime(daily_sum['datetime'])
            fig_daily = px.line(daily_sum, x='date', y='amt', title=f'Daily Spending Over Time - {selected_wallets[0]}',
                               labels={'amt': 'Amount (INR)', 'date': 'Date'})
            figs['daily'] = fig_daily
        except:
            st.write("Not enough data for daily chart.")

        # Time of day bar chart
        time_sum = df.groupby('time_of_day')['amt'].sum().reset_index()
        fig_time = px.bar(time_sum, x='time_of_day', y='amt', title=f'Spending by Time of Day - {selected_wallets[0]}',
                         labels={'amt': 'Amount (INR)', 'time_of_day': 'Time of Day'},
                         category_orders={'time_of_day': ['Early Morning (1-6 AM)', 'Morning (6 AM-12 PM)', 'Afternoon (12-6 PM)', 'Evening (6-10 PM)', 'Late Night (10 PM-1 AM)']})
        figs['time'] = fig_time

        # Weekend vs Weekday pie
        weekend_sum = df.groupby('is_weekend')['amt'].sum().reset_index()
        weekend_sum['day_type'] = weekend_sum['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        fig_week = px.pie(weekend_sum, values='amt', names='day_type', title=f'Weekend vs Weekday Spending - {selected_wallets[0]}')
        figs['week'] = fig_week

        # Heatmap for weekday vs hour spending
        heatmap_data = df.pivot_table(values='amt', index='weekday', columns='hour', aggfunc='sum', fill_value=0)
        # Map weekday numbers to names
        weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        heatmap_data.index = heatmap_data.index.map(weekday_names)
        fig_heatmap = px.imshow(heatmap_data,
                               labels=dict(x="Hour of Day", y="Weekday", color="Amount (INR)"),
                               title="Spending Heatmap: Weekdays vs Time of Day",
                               color_continuous_scale='Blues')
        figs['heatmap'] = fig_heatmap

        # Payee frequency analysis - this should be available since insights is in scope
        payee_df = pd.DataFrame(df.groupby('payee')['amt'].agg(['sum', 'mean', 'count']).reset_index())
        payee_df['percentage'] = (payee_df['count'] / payee_df['count'].sum() * 100).round(1)
        payee_df = payee_df.sort_values('count', ascending=False).head(10)  # Top 10 for visualization

        # Add category information to payees
        payee_category_map = df.drop_duplicates('payee')[['payee', 'mcc_category']].set_index('payee')
        payee_df = payee_df.merge(payee_category_map, left_on='payee', right_index=True, how='left')

        fig_payee_freq = px.bar(payee_df, x='payee', y='count',
                               title='Frequent Payees by Transaction Count',
                               labels={'count': 'Transaction Count', 'payee': 'Payee'},
                               color='mcc_category',
                               text=payee_df['percentage'].astype(str) + '% of all transactions')
        fig_payee_freq.update_traces(textposition='outside')
        figs['payee_freq'] = fig_payee_freq

    else:
        # Multi-user comparative plots

        # 1. Total spending by user
        user_spending = df.groupby('WalletId')['amt'].sum().reset_index()
        fig_user_spending = px.bar(user_spending, x='WalletId', y='amt',
                                  title='Total Spending by User',
                                  labels={'amt': 'Amount (INR)', 'WalletId': 'User ID'},
                                  color='WalletId', color_discrete_map=wallet_color_map)
        figs['user_spending'] = fig_user_spending

        # 2. Average transaction amount by user
        user_avg = df.groupby('WalletId')['amt'].mean().reset_index()
        fig_user_avg = px.bar(user_avg, x='WalletId', y='amt',
                             title='Average Transaction Amount by User',
                             labels={'amt': 'Average Amount (INR)', 'WalletId': 'User ID'},
                             color='WalletId', color_discrete_map=wallet_color_map)
        figs['user_avg'] = fig_user_avg

        # 3. Transaction count by user
        user_count = df.groupby('WalletId')['amt'].count().reset_index()
        fig_user_count = px.bar(user_count, x='WalletId', y='amt',
                               title='Transaction Count by User',
                               labels={'amt': 'Transaction Count', 'WalletId': 'User ID'},
                               color='WalletId', color_discrete_map=wallet_color_map)
        figs['user_count'] = fig_user_count

        # 4. Category spending by user (stacked bar)
        if len(df['mcc_category'].unique()) <= 10:  # Limit categories for readability
            cat_by_user = df.groupby(['WalletId', 'mcc_category'])['amt'].sum().reset_index()
            fig_cat_by_user = px.bar(cat_by_user, x='WalletId', y='amt', color='mcc_category',
                                    title='Category Spending by User',
                                    labels={'amt': 'Amount (INR)', 'WalletId': 'User ID', 'mcc_category': 'Category'},
                                    color_discrete_sequence=px.colors.qualitative.Set3)
            figs['cat_by_user'] = fig_cat_by_user

        # 5. Time of day spending by user
        time_by_user = df.groupby(['WalletId', 'time_of_day'])['amt'].sum().reset_index()
        fig_time_by_user = px.bar(time_by_user, x='time_of_day', y='amt', color='WalletId',
                                 title='Time of Day Spending by User',
                                 labels={'amt': 'Amount (INR)', 'time_of_day': 'Time of Day', 'WalletId': 'User ID'},
                                 color_discrete_map=wallet_color_map, barmode='group',
                                 category_orders={'time_of_day': ['Early Morning (1-6 AM)', 'Morning (6 AM-12 PM)', 'Afternoon (12-6 PM)', 'Evening (6-10 PM)', 'Late Night (10 PM-1 AM)']})
        figs['time_by_user'] = fig_time_by_user

        # 6. Weekend vs Weekday by user
        weekend_by_user = df.groupby(['WalletId', 'is_weekend'])['amt'].sum().reset_index()
        weekend_by_user['day_type'] = weekend_by_user['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        fig_weekend_by_user = px.bar(weekend_by_user, x='WalletId', y='amt', color='day_type',
                                    title='Weekend vs Weekday Spending by User',
                                    labels={'amt': 'Amount (INR)', 'WalletId': 'User ID', 'day_type': 'Day Type'},
                                    color_discrete_sequence=['#FF9999', '#66B2FF'], barmode='group')
        figs['weekend_by_user'] = fig_weekend_by_user

        # 7. Daily spending trends by user (if enough data)
        try:
            df_temp = df.copy()
            df_temp['datetime'] = pd.to_datetime(df_temp['date'] + ' ' + df_temp['time'], dayfirst=True, errors='coerce')
            daily_by_user = df_temp.groupby(['WalletId', df_temp['datetime'].dt.date])['amt'].sum().reset_index()
            daily_by_user['date'] = pd.to_datetime(daily_by_user['datetime'])
            daily_by_user = daily_by_user.sort_values('date')

            fig_daily_by_user = px.line(daily_by_user, x='date', y='amt', color='WalletId',
                                       title='Daily Spending Trends by User',
                                       labels={'amt': 'Amount (INR)', 'date': 'Date', 'WalletId': 'User ID'},
                                       color_discrete_map=wallet_color_map)
            figs['daily_by_user'] = fig_daily_by_user
        except Exception as e:
            pass  # Skip if not enough daily data

    return figs

# Streamlit app
def main():
    st.title("Behavioral Insights from Transaction Data 📊")

    # Load data
    df = load_data()

    # Filter by Wallet ID - select one ID
    wallet_ids = df['WalletId'].unique()
    selected_wallet = st.sidebar.selectbox("Select Wallet ID:", wallet_ids, index=0)
    selected_wallets = [selected_wallet]

    # Filter by date range - using separate From and To date columns
    min_date = pd.to_datetime(df['date_only']).min()
    max_date = pd.to_datetime(df['date_only']).max()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        from_date = st.date_input("From Date:", min_date.date(), min_value=min_date.date(), max_value=max_date.date())
    with col2:
        to_date = st.date_input("To Date:", max_date.date(), min_value=min_date.date(), max_value=max_date.date())

    df_filtered = df[(df['WalletId'].isin(selected_wallets)) &
                   (pd.to_datetime(df['date_only']) >= pd.to_datetime(from_date)) &
                   (pd.to_datetime(df['date_only']) <= pd.to_datetime(to_date))].copy()
    with st.expander("View Filtered Data"):
        st.write(df_filtered)
    # Perform EDA
    insights, df_filtered = perform_eda(df_filtered)

    # Display insights
    if len(selected_wallets) == 1:
        # Single user metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", insights['total_transactions'])
        with col2:
            st.metric("Total Spending", f"₹ {insights['total_spending']}")
        with col3:
            st.metric("Average Transaction", f"₹ {insights['avg_transaction']:.2f}")
        with col4:
            if '50%' in insights['spending_distribution']:
                st.metric("Median Transaction", f"₹ {insights['spending_distribution']['50%']:.2f}")

        # Frequency metrics
        st.markdown("### Transaction Frequency")
        freq_col1, freq_col2, freq_col3 = st.columns(3)
        with freq_col1:
            if insights['hourly_transaction_frequency']:
                peak_hour = max(insights['hourly_transaction_frequency'], key=lambda x: x['transaction_count'])
                st.metric("Peak Hour", f"{int(peak_hour['hour'])}:00 ({peak_hour['transaction_count']} txns)")
        with freq_col2:
            if insights['weekday_transaction_frequency']:
                peak_day = max(insights['weekday_transaction_frequency'], key=lambda x: x['transaction_count'])
                weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                st.metric("Peak Day", f"{weekday_names[peak_day['weekday']]} ({peak_day['transaction_count']} txns)")
        with freq_col3:
            avg_daily_txns = insights['total_transactions'] / len(insights['daily_transaction_frequency']) if insights['daily_transaction_frequency'] else 0
            st.metric("Avg Daily Transactions", f"{avg_daily_txns:.1f}")

        # # Time Series Analysis Metrics
        # st.markdown("### 📊 Time Series Analysis")
        # ts_col1, ts_col2, ts_col3, ts_col4 = st.columns(4)
        # with ts_col1:
        #     trend_direction = "📈 Increasing" if insights['spending_trend_slope'] > 0 else "📉 Decreasing" if insights['spending_trend_slope'] < 0 else "➡️ Stable"
        #     st.metric("Trend Direction", trend_direction)
        # with ts_col2:
        #     st.metric("Avg Daily Spend", f"₹ {insights['avg_daily_spend']:.0f}")
        # with ts_col3:
        #     st.metric("Spending Variance", f"₹ {insights['spending_variance']:.0f}")
        # with ts_col4:
        #     st.metric("Spending Std Dev", f"₹ {insights['spending_std']:.0f}")
    else:
        # Multi-user comparison
        st.markdown("### User Comparison Overview")
        user_comparison_df = pd.DataFrame(insights['user_comparison'])
        st.dataframe(user_comparison_df.style.highlight_max(axis=0), width='stretch')

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", len(selected_wallets))
        with col2:
            st.metric("Combined Spending", f"₹ {insights['total_spending']}")
        with col3:
            st.metric("Combined Transactions", insights['total_transactions'])
        with col4:
            avg_user_spending = user_comparison_df['total_spending'].mean()
            st.metric("Avg User Spending", f"₹ {avg_user_spending:.0f}")

    # # Insights text
    # st.markdown("## Insights Summary")
    # txt = []
    # txt.append("### Spending Patterns & Financial Personality\n")
    # txt.append(f"- Total Transactions: {insights['user_transactions']}\n")
    # txt.append(f"- Total Spending: ₹ {insights['user_total_spending']:.2f}\n")
    # txt.append(f"- Average Transaction: ₹ {insights['user_avg_transaction']:.2f}\n")
    # if insights['spending_distribution']:
    #     txt.append(f"- Median Transaction: ₹ {insights['spending_distribution']['50%']:.2f}\n")
    # txt.append("- Frequency: High-frequency purchases indicate habitual spending.\n")

    # if 'sum' in insights['time_spending']:
    #     sum_dict = insights['time_spending']['sum']
    #     if sum_dict:
    #         max_time = max(sum_dict, key=sum_dict.get)
    #         txt.append(f"- Most active period: {max_time} with ₹ {sum_dict[max_time]:.2f} spent.\n")

    # if insights['weekday_vs_weekend']:
    #     weekend_dict = insights['weekday_vs_weekend'].get(True, {'sum': 0})
    #     weekday_dict = insights['weekday_vs_weekend'].get(False, {'sum': 0})
    #     weekend_spent = weekend_dict['sum']
    #     weekday_spent = weekday_dict['sum']
    #     if weekend_spent > weekday_spent:
    #         txt.append("- Weekend spikes suggest leisure or social spending.\n")
    #         txt.append("\n**Suggestions:**\n1. Offer weekend entertainment discounts.\n2. Promote social event related services.\n")
    #     else:
    #         txt.append("- Consistent spending throughout the week.\n")
    #         txt.append("\n**Suggestions:**\n1. Provide daily convenience services.\n2. Suggest weekly banking tools.\n")

    # if insights['categories']:
    #     txt.append("### Category Preferences (via MCC Codes)\n")
    #     sum_cats = insights['categories']['sum']
    #     top_cats = sorted(sum_cats, key=sum_cats.get, reverse=True)[:5]
    #     for cat in top_cats:
    #         txt.append(f"- {cat}: ₹ {sum_cats[cat]:.2f}\n")
    #     txt.append("- Lifestyle Mapping: Frequent purchases in these categories indicate priorities.\n")
    #     txt.append("\n**Suggestions:**\n1. Offer category-specific loyalty programs.\n2. Recommend related products based on top categories.\n")

    # # More sections
    # txt.append("### Transaction Type Analysis\n")
    # txt.append("- All transactions appear to be card-based electronic payments.\n")
    # txt.append("- Indicates comfort with digital payments.\n")
    # txt.append("\n**Suggestions:**\n1. Provide contactless payment features.\n2. Offer digital wallet enhancements.\n")

    # txt.append("### Seasonality & Life Events\n")
    # try:
    #     monthly = df_filtered.groupby(df_filtered['datetime'].dt.month)['amt'].sum()
    #     peak_month = monthly.idxmax()
    #     peak_amount = monthly.max()
    #     txt.append(f"- Peak spending in month {peak_month} with ₹ {peak_amount:.2f}.\n")
    #     txt.append("\n**Suggestions:**\n1. Plan seasonal promotions.\n2. Develop month-specific financial products.\n")
    # except:
    #     txt.append("- Insufficient data for seasonality analysis.\n")
    #     txt.append("\n**Suggestions:**\n1. Encourage longer transaction history for better insights.\n")

    # txt.append("### Behavioral Segmentation\n")
    # avg_amt = insights['user_avg_transaction']
    # if avg_amt < 10:
    #     segment = "Planner - Budget-conscious, regular small purchases"
    # elif 10 <= avg_amt < 50:
    #     segment = "Explorer - Varied spending with moderate amounts"
    # else:
    #     segment = "Status Seeker - High-value purchases"
    # txt.append(f"- Suggested Segment: {segment}\n")

    # # Advanced techniques
    # txt.append("### Advanced Techniques\n")
    # txt.append("- Time Series Analysis: Possible seasonality detected.\n")
    # txt.append("- Potential future income prediction based on spending consistency.\n")

    # # Profit strategies
    # txt.append("### Profit Strategies from Behavioral Spend Data\n")
    # txt.append("#### Personalized Marketing & Offers\n")
    # if insights['categories']:
    #     top_cats = sorted(insights['categories']['sum'], key=insights['categories']['sum'].get, reverse=True)[:2]
    #     txt.append(f"- Target high-frequency buyers in top categories: {', '.join(top_cats)}\n")

    # st.write(''.join(txt))

    # Plots
    figs = create_plots(df_filtered, selected_wallets)
    if figs:
        st.markdown("## Visualizations")

        # Category spending section with tabs
        st.markdown("## 🎯 Category Spending Analysis")

        if len(selected_wallets) == 1:
            # Single user category analysis
            tab1, tab2 = st.tabs(["📊 Category Details", "🌀 Range Analysis"])

            with tab1:
                # Create bar chart for mcc_category with percentages
                cat_data = df_filtered.groupby('mcc_category')['amt'].sum().reset_index()
                total_spending = cat_data['amt'].sum()
                cat_data['percentage'] = (cat_data['amt'] / total_spending * 100).round(1)
                cat_data = cat_data.sort_values('amt', ascending=False)

                # Left side: Chart, Right side: Analysis tabs
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    fig_cat_pct = px.bar(cat_data, x='mcc_category', y='amt',
                                        title='Spending by MCC Category',
                                        labels={'amt': 'Amount (INR)', 'mcc_category': 'Category'},
                                        text=cat_data['percentage'].astype(str) + '%')
                    fig_cat_pct.update_traces(textposition='outside')
                    st.plotly_chart(fig_cat_pct, use_container_width=True)

                with col_right:
                    analysis_tab1, analysis_tab2 = st.tabs(["📈 Rankings", "Remarks"])

                    with analysis_tab1:
                        st.markdown("### Complete Category Distribution")
                        st.markdown("**All Categories - Amount & Percentage**")

                        # Display as a clean table
                        table_data = cat_data[['mcc_category', 'amt', 'percentage']].copy()
                        table_data.columns = ['Category', 'Amount (₹)', 'Percentage (%)']
                        table_data['Amount (₹)'] = table_data['Amount (₹)'].astype(int)
                        st.dataframe(table_data, width='stretch')

                    with analysis_tab2:
                        st.markdown("### 💰 Notes")
                        total_amt = df_filtered['amt'].sum()

                        suggestions = [
                            "🎯 **Target High-Value Categories**: Focus marketing on top spending categories",
                            "💳 **Loyalty Programs**: Reward customers in their preferred spending categories",
                            "📈 **Upsell Opportunities**: Promote related products within popular categories",
                            "🎪 **Seasonal Promotions**: Time offers based on peak category spending periods",
                            "🤝 **Partner Programs**: Collaborate with merchants in high-spend categories"
                        ]

                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")

            with tab2:
                # Left side: Pie Chart for mcc_category_range
                range_data = df_filtered.groupby('mcc_category_range')['amt'].sum().reset_index()
                total_spending = range_data['amt'].sum()
                range_data['percentage'] = (range_data['amt'] / total_spending * 100).round(1)
                range_data = range_data.sort_values('amt', ascending=False)  # Sort by amount

                # Left-right split for Range Analysis
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    fig_pie = px.pie(range_data, values='amt', names='mcc_category_range',
                                    title='Spending by MCC Range Categories',
                                    labels={'mcc_category_range': 'Range Category'},
                                    hover_data=['percentage'],
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_right:
                    range_analysis_tab1, range_analysis_tab2 = st.tabs(["📈 Range Rankings", "💡 Remarks"])

                    with range_analysis_tab1:
                        st.markdown("### Complete Range Distribution")
                        st.markdown("**All Range Categories - Amount & Percentage**")

                        # Display as a clean table
                        table_data = range_data[['mcc_category_range', 'amt', 'percentage']].copy()
                        table_data.columns = ['Range Category', 'Amount (₹)', 'Percentage (%)']
                        table_data['Amount (₹)'] = table_data['Amount (₹)'].astype(int)
                        st.dataframe(table_data, width='stretch')

                    with range_analysis_tab2:
                        st.markdown("### Notes")

                        # Find dominant range category
                        dominant_range = range_data.iloc[0]['mcc_category_range'] if len(range_data) > 0 else "N/A"
                        dominant_percentage = range_data.iloc[0]['percentage'] if len(range_data) > 0 else 0

                        suggestions = [
                            "🎯 **Focus on Dominant Range Categories**: Target marketing campaigns at the most popular MCC ranges",
                            f"🏆 **Emphasize {dominant_range}** ({dominant_percentage:.1f}% of spending) for strategic partnerships",
                            "📊 **Range-Based Segmentation**: Create customer segments based on MCC range preferences",
                            "🎪 **Range-Specific Promotions**: Design promotional offers tailored to range-based spending patterns",
                            "🤝 **Merchant Network Expansion**: Strengthen relationships with merchants in high-spend ranges",
                            "💳 **Range-Optimized Products**: Develop financial products catering to specific MCC range behaviors"
                        ]

                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")

        else:
            # Multi-user comparative plots

            # User metrics charts
            col1, col2 = st.columns(2)
            with col1:
                if 'user_spending' in figs:
                    st.markdown("### Total Spending by User")
                    st.plotly_chart(figs['user_spending'])
            with col2:
                if 'user_avg' in figs:
                    st.markdown("### Average Transaction by User")
                    st.plotly_chart(figs['user_avg'])

            col1, col2 = st.columns(2)
            with col1:
                if 'user_count' in figs:
                    st.markdown("### Transaction Count by User")
                    st.plotly_chart(figs['user_count'])
            with col2:
                if 'weekend_by_user' in figs:
                    st.markdown("### Weekend vs Weekday by User")
                    st.plotly_chart(figs['weekend_by_user'])

            # Category analysis tabs for multi-user
            st.markdown("### 🎯 Multi-User Category Analysis")
            cat_tab1, cat_tab2 = st.tabs(["📊 Category Bar Chart", "🫧 Range Pie Chart"])

            with cat_tab1:
                # Multi-user category bar chart
                cat_by_user = df_filtered.groupby(['WalletId', 'mcc_category'])['amt'].sum().reset_index()
                total_by_user = cat_by_user.groupby('WalletId')['amt'].sum().reset_index().rename(columns={'amt': 'total'})

                # Add percentage calculation
                cat_by_user = cat_by_user.merge(total_by_user, on='WalletId')
                cat_by_user['percentage'] = (cat_by_user['amt'] / cat_by_user['total'] * 100).round(1)

                wallet_color_map = {wallet: f'hsl({i*360/len(selected_wallets)}, 70%, 50%)' for i, wallet in enumerate(selected_wallets)}
                fig_multi_cat = px.bar(cat_by_user, x='WalletId', y='amt', color='mcc_category',
                                      title='Category Spending by User',
                                      labels={'amt': 'Amount (INR)', 'WalletId': 'User ID', 'mcc_category': 'Category'},
                                      color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_multi_cat, use_container_width=True)

            with cat_tab2:
                # Multi-user range pie chart (showing combined ranges)
                range_combined = df_filtered.groupby('mcc_category_range')['amt'].sum().reset_index()
                total_spending = range_combined['amt'].sum()
                range_combined['percentage'] = (range_combined['amt'] / total_spending * 100).round(1)

                fig_multi_pie = px.pie(range_combined, values='amt', names='mcc_category_range',
                                      title='Combined Range Category Spending',
                                      labels={'mcc_category_range': 'Range Category'},
                                      hover_data=['percentage'])
                fig_multi_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_multi_pie, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if 'time_by_user' in figs:
                    st.markdown("### Time of Day by User")
                    st.plotly_chart(figs['time_by_user'])
            with col2:
                if 'daily_by_user' in figs:
                    st.markdown("### Daily Spending Trends by User")
                    st.plotly_chart(figs['daily_by_user'])

        # Other visualizations section
        st.markdown("## 📈 Other Analysis")

        if len(selected_wallets) == 1:
            if 'daily' in figs:
                st.markdown("### Daily Spending Over Time")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(figs['daily'])
                with col2:
                    st.write("The line chart tracks daily spending trends to identify patterns.\n\n**Suggestions:**\n1. Provide budget alerts for high-spend days.\n2. Suggest financial planning tools for consistent spending.")
            if 'time' in figs:
                st.markdown("### Spending by Time of Day")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(figs['time'])
                with col2:
                    st.write("This bar chart illustrates spending by time of day, showing peak activity periods.\n\n**Suggestions:**\n1. Send promotions during peak times.\n2. Optimize customer service hours.")
            if 'week' in figs:
                st.markdown("### Weekend vs Weekday Spending")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(figs['week'])
                with col2:
                    st.write("The pie chart compares spending on weekends and weekdays.\n\n**Suggestions:**\n1. Offer weekend-exclusive deals.\n2. Provide commute or work-related financial services for weekdays.")
            if 'heatmap' in figs:
                st.markdown("### 📊 Customizable Spending Heatmap")

                # Controls for heatmap customization
                heatmap_col1, heatmap_col2, heatmap_col3 = st.columns(3)

                with heatmap_col1:
                    x_axis = st.selectbox(
                        "X-Axis (Columns)",
                        ["Hour", "Weekday", "Month", "Day"],
                        index=0,
                        help="Choose the dimension for X-axis"
                    )

                with heatmap_col2:
                    y_axis = st.selectbox(
                        "Y-Axis (Rows)",
                        ["Weekday", "Hour", "Month", "Day"],
                        index=0 if x_axis != "Weekday" else 1,
                        help="Choose the dimension for Y-axis"
                    )

                with heatmap_col3:
                    value_type = st.selectbox(
                        "Value Type",
                        ["Transaction Amount", "Transaction Count"],
                        index=0,
                        help="Choose whether to show amount or transaction frequency"
                    )

                # Prevent same axis selection
                if x_axis == y_axis:
                    st.error("Please select different dimensions for X and Y axes.")
                else:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Generate custom heatmap based on selections
                        heatmap_data = df_filtered.copy()

                        # Map dimensions to column names
                        dim_map = {
                            "Hour": "hour",
                            "Weekday": "weekday",
                            "Month": "month",
                            "Day": "day"
                        }

                        # Add month and day columns if not present
                        if x_axis == "Month" or y_axis == "Month" or x_axis == "Day" or y_axis == "Day":
                            heatmap_data['month'] = pd.to_datetime(heatmap_data['date_only']).dt.month
                            heatmap_data['day'] = pd.to_datetime(heatmap_data['date_only']).dt.day

                        x_col = dim_map[x_axis]
                        y_col = dim_map[y_axis]

                        # Create pivot table
                        if value_type == "Transaction Amount":
                            heatmap_pivot = heatmap_data.pivot_table(
                                values='amt', index=y_col, columns=x_col,
                                aggfunc='sum', fill_value=0
                            )
                        else:  # Transaction Count
                            heatmap_pivot = heatmap_data.pivot_table(
                                values='amt', index=y_col, columns=x_col,
                                aggfunc='count', fill_value=0
                            )

                        # Customize labels
                        x_labels = {}
                        y_labels = {}

                        if x_axis == "Weekday":
                            x_labels = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
                        if y_axis == "Weekday":
                            y_labels = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

                        if x_axis == "Month":
                            x_labels = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
                        if y_axis == "Month":
                            y_labels = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

                        # Apply labels to index and columns
                        if y_labels:
                            heatmap_pivot.index = heatmap_pivot.index.map(lambda x: y_labels.get(x, x))
                        if x_labels:
                            heatmap_pivot.columns = heatmap_pivot.columns.map(lambda x: x_labels.get(x, x))

                        # Create heatmap
                        title_suffix = f"{y_axis} vs {x_axis} ({value_type})"
                        fig_custom_heatmap = px.imshow(
                            heatmap_pivot,
                            labels=dict(x=x_axis, y=y_axis, color=f"{'Amount' if value_type == 'Transaction Amount' else 'Count'}"),
                            title=f"Spending Heatmap: {title_suffix}",
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_custom_heatmap, use_container_width=True)

                    with col2:
                        # Dynamic insights based on selection
                        insights_text = f"The heatmap shows {value_type.lower()} patterns across {y_axis.lower()} and {x_axis.lower()}."

                        if x_axis == "Hour" and y_axis == "Weekday":
                            insights_text += "\n\n**Weekly Rhythm Insights:**\n• Identifies peak hours on different days\n• Helps optimize customer service timing\n• Reveals daily behavioral patterns"
                        elif x_axis == "Month" and y_axis == "Weekday":
                            insights_text += "\n\n**Seasonal Weekly Patterns:**\n• Shows how spending varies by weekday across months\n• Helps plan monthly promotions\n• Reveals seasonal behavioral shifts"
                        elif x_axis == "Day" and y_axis == "Month":
                            insights_text += "\n\n**Monthly Spending Calendar:**\n• Shows spending distribution throughout each month\n• Identifies peak spending days\n• Helps with payroll and bill timing"
                        else:
                            insights_text += "\n\n**Calculation:**\n• **Amount**: Shows total money spent in INR\n• **Count**: Shows number of transactions\n\nDarker colors indicate higher values."

                        st.write(insights_text)

                        # Show some summary stats
                        if value_type == "Transaction Amount":
                            st.markdown("**Summary Statistics:**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Peak Value", f"₹ {heatmap_pivot.max().max():.0f}")
                            with col_b:
                                st.metric("Average Value", f"₹ {heatmap_pivot.mean().mean():.0f}")
                            with col_c:
                                st.metric("Total Sum", f"₹ {heatmap_pivot.sum().sum():.0f}")
                        else:
                            st.markdown("**Summary Statistics:**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Peak Count", f"{heatmap_pivot.max().max():.0f}")
                            with col_b:
                                st.metric("Average Count", f"{heatmap_pivot.mean().mean():.1f}")
                            with col_c:
                                st.metric("Total Count", f"{heatmap_pivot.sum().sum():.0f}")
            if 'payee_freq' in figs:
                st.markdown("### 🏪 Payee Analysis")

                # Selection button for different percentage bases
                analysis_type = st.radio(
                    "Select analysis basis:",
                    ["Transaction Frequency", "Amount Spent", "Weekday Frequency", "Monthly Frequency"],
                    horizontal=True,
                    help="Choose how to analyze payee percentages"
                )

                col1, col2 = st.columns(2)

                if analysis_type == "Transaction Frequency":
                    # Show payee frequency table
                    payee_data = pd.DataFrame(df_filtered.groupby('payee')['amt'].agg(['sum', 'count', 'mean']).reset_index())
                    payee_data['percentage'] = (payee_data['count'] / payee_data['count'].sum() * 100).round(1)
                    payee_data = payee_data.sort_values('count', ascending=False).head(15)

                    # Add category information
                    payee_category_map = df_filtered.drop_duplicates('payee')[['payee', 'mcc_category']].set_index('payee')
                    payee_data = payee_data.merge(payee_category_map, left_on='payee', right_index=True, how='left')

                    with col1:
                        fig_payee_freq = px.bar(payee_data, x='payee', y='count',
                                               title='Frequent Payees by Transaction Count',
                                               labels={'count': 'Transaction Count', 'payee': 'Payee'},
                                               color='mcc_category',
                                               text=payee_data['percentage'].astype(str) + '% of all transactions')
                        fig_payee_freq.update_traces(textposition='outside')
                        st.plotly_chart(fig_payee_freq, use_container_width=True)

                    with col2:
                        st.write("Top merchants by transaction frequency, colored by category. This shows where customers shop most often.\n\n**Business insights include:**\n1. Major merchant relationships and loyalty patterns\n2. Category preferences through merchant choices\n3. Opportunities for merchant partnerships\n4. Digital payment adoption in different merchant types\n\n**Percentage**: Based on total transaction count")

                        # Show payee frequency table
                        payee_data.columns = ['Merchant', 'Total Amount (₹)', 'Transaction Count', 'Avg Amount (₹)', 'Frequency %', 'Category']
                        payee_data['Total Amount (₹)'] = payee_data['Total Amount (₹)'].astype(int)
                        payee_data['Avg Amount (₹)'] = payee_data['Avg Amount (₹)'].round(0).astype(int)
                        st.dataframe(payee_data[['Merchant', 'Category', 'Transaction Count', 'Total Amount (₹)', 'Avg Amount (₹)', 'Frequency %']], width='stretch')

                elif analysis_type == "Amount Spent":
                    # Show payee by amount spent
                    payee_data = pd.DataFrame(df_filtered.groupby('payee')['amt'].agg(['sum', 'count', 'mean']).reset_index())
                    payee_data['percentage'] = (payee_data['sum'] / payee_data['sum'].sum() * 100).round(1)
                    payee_data = payee_data.sort_values('sum', ascending=False).head(15)

                    # Add category information
                    payee_category_map = df_filtered.drop_duplicates('payee')[['payee', 'mcc_category']].set_index('payee')
                    payee_data = payee_data.merge(payee_category_map, left_on='payee', right_index=True, how='left')

                    with col1:
                        fig_payee_amount = px.bar(payee_data, x='payee', y='sum',
                                                 title='Top Payees by Amount Spent',
                                                 labels={'sum': 'Total Amount (INR)', 'payee': 'Payee'},
                                                 color='mcc_category',
                                                 text=payee_data['percentage'].astype(str) + '% of total spending')
                        fig_payee_amount.update_traces(textposition='outside')
                        st.plotly_chart(fig_payee_amount, use_container_width=True)

                    with col2:
                        st.write("Top merchants by total amount spent, colored by category. Shows the most significant financial relationships.\n\n**Business insights include:**\n1. High-value merchant partnerships\n2. Category spending dominance\n3. Premium customer segments\n4. Loyalty program optimization\n\n**Percentage**: Based on total amount spent")

                        # Show payee amount table
                        payee_data.columns = ['Merchant', 'Total Amount (₹)', 'Transaction Count', 'Avg Amount (₹)', 'Amount %', 'Category']
                        payee_data['Total Amount (₹)'] = payee_data['Total Amount (₹)'].astype(int)
                        payee_data['Avg Amount (₹)'] = payee_data['Avg Amount (₹)'].round(0).astype(int)
                        st.dataframe(payee_data[['Merchant', 'Category', 'Total Amount (₹)', 'Transaction Count', 'Avg Amount (₹)', 'Amount %']], width='stretch')

                elif analysis_type == "Weekday Frequency":
                    # Get weekday frequencies for payees
                    weekday_payee = df_filtered.groupby(['payee', 'weekday'])['amt'].count().reset_index()
                    weekday_payee = weekday_payee.pivot(index='payee', columns='weekday', values='amt').fillna(0)
                    weekday_payee['total_txns'] = weekday_payee.sum(axis=1)

                    # Calculate percentage for each weekday
                    for col in range(7):
                        weekday_payee[f'weekday_{col}_pct'] = (weekday_payee[col] / weekday_payee[col].sum() * 100).round(1)

                    # Get top payees and show weekday pattern
                    top_payees = weekday_payee.nlargest(10, 'total_txns')
                    payee_weekday_data = []
                    for payee in top_payees.index:
                        for wd in range(7):
                            payee_weekday_data.append({
                                'Payee': payee,
                                'Weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][wd],
                                'Transactions': int(top_payees.loc[payee, wd]),
                                'Percentage': top_payees.loc[payee, f'weekday_{wd}_pct']
                            })

                    weekday_df = pd.DataFrame(payee_weekday_data)

                    with col1:
                        fig_weekday = px.bar(weekday_df, x='Payee', y='Transactions', color='Weekday',
                                           title='Payee Transaction Patterns by Weekday',
                                           labels={'Transactions': 'Transaction Count'},
                                           barmode='stack')
                        st.plotly_chart(fig_weekday, use_container_width=True)

                    with col2:
                        st.write("Weekday transaction patterns for top merchants. Shows which days different payees are most active.\n\n**Business insights include:**\n1. Optimal timing for merchant promotions\n2. Weekly shopping patterns\n3. Peak days for different merchant categories\n4. Inventory and staffing optimization\n\n**Percentage**: Each weekday's contribution to total transactions")

                        # Show weekday table
                        weekday_pivot = weekday_df.pivot(index='Payee', columns='Weekday', values=['Transactions', 'Percentage'])
                        weekday_pivot.columns = [f'{col[0]} ({col[1]})' for col in weekday_pivot.columns]
                        st.dataframe(weekday_pivot, width='stretch')

                elif analysis_type == "Monthly Frequency":
                    # Get monthly frequencies for payees
                    monthly_payee = df_filtered.copy()
                    monthly_payee['month'] = pd.to_datetime(monthly_payee['date_only']).dt.month
                    monthly_payee = monthly_payee.groupby(['payee', 'month'])['amt'].count().reset_index()
                    monthly_payee = monthly_payee.pivot(index='payee', columns='month', values='amt').fillna(0)
                    monthly_payee['total_txns'] = monthly_payee.sum(axis=1)

                    # Calculate percentage for each month
                    for month in range(1, 13):
                        monthly_payee[f'month_{month}_pct'] = (monthly_payee.get(month, 0) / monthly_payee[monthly_payee.columns[:-1]].sum().sum() * 100).round(1) if month in monthly_payee.columns else 0

                    # Get top payees and show monthly pattern
                    top_payees = monthly_payee.nlargest(10, 'total_txns')
                    payee_monthly_data = []
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                    for payee in top_payees.index:
                        for i, month in enumerate(range(1, 13)):
                            if month in top_payees.columns:
                                payee_monthly_data.append({
                                    'Payee': payee,
                                    'Month': month_names[i],
                                    'Transactions': int(top_payees.loc[payee, month]),
                                    'Percentage': top_payees.loc[payee, f'month_{month}_pct'] if f'month_{month}_pct' in top_payees.columns else 0
                                })

                    monthly_df = pd.DataFrame(payee_monthly_data)

                    with col1:
                        fig_monthly = px.bar(monthly_df, x='Payee', y='Transactions', color='Month',
                                           title='Payee Transaction Patterns by Month',
                                           labels={'Transactions': 'Transaction Count'},
                                           barmode='stack')
                        st.plotly_chart(fig_monthly, use_container_width=True)

                    with col2:
                        st.write("Monthly transaction patterns for top merchants. Shows seasonal shopping behavior.\n\n**Business insights include:**\n1. Seasonal merchant category trends\n2. Holiday shopping patterns\n3. Monthly budget planning\n4. Inventory planning by season\n\n**Percentage**: Each month's contribution to total transactions")

                        # Show monthly table
                        try:
                            monthly_pivot = monthly_df.pivot(index='Payee', columns='Month', values=['Transactions', 'Percentage'])
                            monthly_pivot.columns = [f'{col[0]} ({col[1]})' for col in monthly_pivot.columns]
                            st.dataframe(monthly_pivot, width='stretch')
                        except:
                            st.write("Not enough monthly data to display detailed patterns.")

        # Time Series Analysis Section
        # if len(selected_wallets) == 1 and insights['monthly_spending_trend']:
        #     st.markdown("## ⏰ Time Series Analysis")

        #     # Monthly spending trend and seasonal patterns
        #     col1, col2 = st.columns(2)

        #     # Monthly trend
        #     with col1:
        #         monthly_df = pd.DataFrame(insights['monthly_spending_trend'])
        #         if len(monthly_df) > 0:
        #             monthly_df['month'] = pd.to_datetime(monthly_df['month'])
        #             fig_monthly = px.line(monthly_df, x='month', y='amt',
        #                                  title='Monthly Spending Trend',
        #                                  labels={'amt': 'Amount (INR)', 'month': 'Month'},
        #                                  markers=True)
        #             st.plotly_chart(fig_monthly, use_container_width=True)

        #     # Seasonal monthly patterns
        #     with col2:
        #         seasonal_df = pd.DataFrame(insights['seasonal_monthly_patterns'])
        #         if len(seasonal_df) > 0:
        #             fig_seasonal = px.bar(seasonal_df, x='month_name', y='sum',
        #                                  title='Seasonal Monthly Spending Patterns',
        #                                  labels={'sum': 'Total Amount (INR)', 'month_name': 'Month'},
        #                                  color='sum', color_continuous_scale='Viridis')
        #             st.plotly_chart(fig_seasonal, use_container_width=True)

        #     # Weekly trends and moving averages
        #     if insights['weekly_spending_trend']:
        #         st.markdown("### Weekly Spending Trends")
        #         weekly_df = pd.DataFrame(insights['weekly_spending_trend'])
        #         if len(weekly_df) > 0:
        #             weekly_df['week'] = pd.to_datetime(weekly_df['week'])
        #             fig_weekly = px.line(weekly_df, x='week', y='amt',
        #                                title='Weekly Spending Trends',
        #                                labels={'amt': 'Amount (INR)', 'week': 'Week'},
        #                                markers=True)
        #             st.plotly_chart(fig_weekly, use_container_width=True)

        #             # Moving average if available
        #             if insights['daily_moving_average']:
        #                 st.markdown("### 7-Day Moving Average")
        #                 ma_df = pd.DataFrame(insights['daily_moving_average'])
        #                 if len(ma_df) > 0:
        #                     ma_df['date_only'] = pd.to_datetime(ma_df['date_only'])
        #                     fig_ma = px.line(ma_df, x='date_only', y='7_day_moving_avg',
        #                                    title='7-Day Moving Average of Daily Spending',
        #                                    labels={'7_day_moving_avg': 'Moving Average (INR)', 'date_only': 'Date'})
        #                     st.plotly_chart(fig_ma, use_container_width=True)

        # st.markdown("---")
        # st.markdown("*Charts automatically adapt based on single vs multiple user selection*")

if __name__ == "__main__":
    main()
