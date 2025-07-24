import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from collections import defaultdict
import random
import calendar
register_matplotlib_converters()

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv('https://raw.githubusercontent.com/freeCodeCamp/boilerplate-page-view-time-series-visualizer/88b7d3d03addcb7ab5d5fde6a1d40598316a06e4/fcc-forum-pageviews.csv')


# Clean data
df = df[
    (df['value'] >= df['value'].quantile(0.025)) &
    (df['value'] <= df['value'].quantile(0.975))
]
print(df.shape)
df['date'] = pd.to_datetime(df['date'])
print(df.dtypes)
print(df.head())


def draw_line_plot():
    # Draw line plot
    fig = plt.figure(figsize=(20, 8))
    plt.plot('date', 'value', data = df)
    plt.title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    plt.xlabel('Date')
    plt.ylabel('Page Views')

    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig


def draw_bar_plot():

    global df

    # Ensure datetime and extract year/month
    df_bar = df.copy()
    df_bar['year'] = df_bar['date'].dt.year
    df_bar['month'] = df_bar['date'].dt.month

    # Step 1: Create nested dictionary of average page views per year-month
    nested_avg = defaultdict(dict)
    for (year, month), group in df_bar.groupby(['year', 'month']):
        nested_avg[year][month] = group['value'].mean()

    # Step 2: Prepare for plotting
    years = sorted(nested_avg.keys())
    months = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in months]

    x = np.arange(len(years))  # x positions for each year group
    width = 0.065
    multiplier = 0

    fig, ax = plt.subplots(figsize=(16, 6), layout='constrained')

    # Step 3: Plot each month's data as grouped bars
    for month in months:
        offset = width * multiplier
        values = [nested_avg[year].get(month, 0) for year in years]
        rects = ax.bar(x + offset, values, width, label=calendar.month_name[month])
        multiplier += 1

    # Step 4: Customize plot
    ax.set_xlabel('Years')
    ax.set_ylabel('Average Page Views')
    ax.set_title('Average Daily Page Views by Month Grouped by Year')
    ax.set_xticks(x + width * 6)  # center tick labels between groups
    ax.set_xticklabels(years)
    ax.legend(title='Months', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Save and return
    fig.savefig('bar_plot.png')
    return fig


def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box['month_num'] = df['date'].dt.month
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box.date]
    df_box['month'] = [d.strftime('%b') for d in df_box.date]
    df_box.sort_values('month_num', inplace=True)
    colors = ['#991a21', '#ff595a', '#a7e163', '#ffad00', '#006298', '#d14600', '#f9e547', '#6ed34b', '#ffa38b', '#1ececa', '#741bc1', '#20912d']
    print(df_box.dtypes)
    np.float = float

    # Draw box plots (using Seaborn)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax1 = sns.boxplot(data=df_box, x='year', y='value', ax=ax[0], hue='year', palette=random.sample(colors, k=4))
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Page Views')
    ax1.set_title('Year-wise Box Plot (Trend)')
    ax2 = sns.boxplot(data=df_box, x='month', y='value', ax=ax[1], hue='month', palette=colors)
    ax2.set_title('Month-wise Box Plot (Seasonality)')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Page Views')

    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig
