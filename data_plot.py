import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

metal_data = data[['Gold', 'Silver', 'Palladium', 'Platinum', 'Copper']]
energy_data = data[['WTI Crude Oil', 'Brent Crude Oil', 'Natural Gas']]
agri_data = data[['Corn', 'Cocoa', 'Cotton', 'Coffee', 'Lean Hogs', 'Soybeans']]

# Define your color palettes and markers for the different types of commodities
metal_colors = ['blue', 'yellow', 'green', 'orange', 'royalblue']
energy_colors = ['green', 'yellow', 'royalblue']
agri_colors = ['blue', 'yellow', 'green', 'orange', 'darkturquoise', 'royalblue']
# Markers for the different types of commodities
metal_markers = ['^', '>', 'x', '+', '*']
energy_markers = ['o', 's', 'd']
agri_markers = ['.', 'P', 'H', 'X', 'D', 'p']

# Set consistent plot size, line width, and axis ranges
plot_size = (15, 8)
line_width = 2
x_axis_range = [pd.Timestamp('1990-01-01'), pd.Timestamp('2025-12-31')]
y_axis_range = [-0.5, 0.5]

# def plot_commodities(data, title, colors, markers, ylabel='Price', x_lim=x_axis_range, y_lim=y_axis_range):
#     plt.figure(figsize=plot_size)
#     for i, column in enumerate(data.columns):
#         plt.plot(data.index, data[column],
#                  label=column, linewidth=line_width, color=colors[i],
#                  marker=markers[i], markevery=int(len(data) / 20))  # Adjust markevery based on data length
#     plt.title(title, fontsize=18)
#     plt.xlabel('Date', fontsize=14)
#     plt.ylabel(ylabel, fontsize=14)
#     plt.legend(title='Commodity', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.xlim(x_lim)
#     plt.ylim(y_lim)
#     plt.tight_layout()
#     plt.show()

# plot_commodities(metal_data, 'Metal Commodities', metal_colors, metal_markers)

# plot_commodities(energy_data, 'Energy Commodities', energy_colors, energy_markers)

# plot_commodities(agri_data, 'Agricultural Commodities', agri_colors, agri_markers)




fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 9))  # Adjust total height as needed


def plot_commodities(ax, data, title, colors, markers, ylabel='Return', x_lim=x_axis_range, y_lim=y_axis_range):
    for i, column in enumerate(data.columns):
        ax.plot(data.index, data[column], label=column, linewidth=line_width, color=colors[i],
                marker=markers[i], markevery=int(len(data) / 20))  # Adjust markevery based on data length
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(title='Commodity', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

plot_commodities(axes[0], metal_data, 'Metal Commodities', metal_colors, metal_markers)
plot_commodities(axes[1], energy_data, 'Energy Commodities', energy_colors, energy_markers)
plot_commodities(axes[2], agri_data, 'Agricultural Commodities', agri_colors, agri_markers)

plt.tight_layout()
plt.show()