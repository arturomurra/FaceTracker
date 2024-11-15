import pandas as pd
import matplotlib.pyplot as plt

# Load the rewards CSV file
rewards_df = pd.read_csv('rewards.csv')

# Print column names to ensure they match
print("Column names:", rewards_df.columns)

# Convert to numpy arrays to avoid multi-dimensional indexing issue
epochs = rewards_df['Epoch'].to_numpy()
total_rewards = rewards_df['Total Rewards'].to_numpy()

# Calculate the moving average of rewards (e.g., window size of 10)
window_size = 10
rewards_df['Moving_Avg'] = rewards_df['Total Rewards'].rolling(window=window_size).mean()

# Convert moving average to numpy for plotting
moving_avg = rewards_df['Moving_Avg'].to_numpy()

# Plot the total rewards and moving average
plt.figure(figsize=(10, 6))
plt.plot(epochs, total_rewards, label='Total Rewards')
plt.plot(epochs, moving_avg, label=f'{window_size}-Epoch Moving Average', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Rewards')
plt.title('Rewards and Moving Average Over Time')
plt.legend()

# Save the plot as an image (PNG format)
plot_filename = 'rewards_plot.png'
plt.savefig(plot_filename)

# Show the plot
plt.show()

print(f"Plot saved as {plot_filename}")
