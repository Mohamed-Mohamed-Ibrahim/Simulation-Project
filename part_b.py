import numpy as np
import matplotlib.pyplot as plt

def simulate_queue(λ, μ1, μ2, T, q):
    """
    Simulates a two-server queue system over a given period.

    Parameters:
    λ (float)  : Arrival rate (mean rate of arrivals per unit time).
    μ1 (float) : Service rate of server 1 (mean rate of service completions per unit time).
    μ2 (float) : Service rate of server 2 (mean rate of service completions per unit time).
    T (float)  : Total simulation time.
    q (int)    : Initial number of c"ustomers in queue 1.

    Returns:
    tuple: Two lists containing timestamps and queue lengths for L1 and L2.
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    # Initialize time and queue lengths
    t  = 0
    L1 = q
    L2 = 0

    # Generate interarrival and service times based on exponential distributions
    arrivals  = np.random.exponential(1 / λ, size=int(λ * T * 2))
    services1 = np.random.exponential(1 / μ1, size=int(μ1 * T * 2))
    services2 = np.random.exponential(1 / μ2, size=int(μ2 * T * 2))

    # Initialize event times
    t_arrival  = arrivals[0]
    t_service1 = t + services1[0] if L1 > 0 else float('inf')
    t_service2 = float('inf')

    # Index trackers for the events
    arrival_idx  = 1
    service1_idx = 1
    service2_idx = 0

    # Lists to store queue lengths over time
    L1_times = []
    L2_times = []

    # Run the simulation until time T
    while t < T:
        if t_arrival <= t_service1 and t_arrival <= t_service2:
            # Handle arrival event
            t = t_arrival
            L1 += 1
            L1_times.append((t, L1))
            L2_times.append((t, L2))
            t_arrival = t + arrivals[arrival_idx] if arrival_idx < len(arrivals) else float('inf')
            arrival_idx += 1
            if L1 == 1:
                t_service1 = t + services1[service1_idx]
        elif t_service1 <= t_arrival and t_service1 <= t_service2:
            # Handle service completion at server 1
            t = t_service1
            L1 -= 1
            L2 += 1
            L1_times.append((t, L1))
            L2_times.append((t, L2))
            t_service1 = t + services1[service1_idx] if L1 > 0 else float('inf')
            service1_idx += 1
            if L2 == 1:
                t_service2 = t + services2[service2_idx]
        else:
            # Handle service completion at server 2
            t = t_service2
            L2 -= 1
            L1_times.append((t, L1))
            L2_times.append((t, L2))
            t_service2 = t + services2[service2_idx] if L2 > 0 else float('inf')
            service2_idx += 1

    return L1_times, L2_times

def plot_queues(L1_times, L2_times, title):
    """
    Plots the lengths of two queues over time.

    Parameters:
    L1_times (list): List of tuples with time and queue length for queue 1.
    L2_times (list): List of tuples with time and queue length for queue 2.
    title (str): Title of the plot.
    """
    # Convert lists to numpy arrays for easier plotting
    L1_times = np.array(L1_times)
    L2_times = np.array(L2_times)

    # Plot the queue lengths over time
    plt.plot(L1_times[:, 0], L1_times[:, 1], label='L1 Queue Length')
    plt.plot(L2_times[:, 0], L2_times[:, 1], label='L2 Queue Length')

    # Set plot labels and title
    plt.xlabel('Time')
    plt.ylabel('Queue Length')
    plt.title(title)
    plt.legend()
    plt.grid(True)

# Parameters
λ_values  = [1, 5]  # Arrival rates to simulate
μ1_values = [2, 4]  # Service rates for server 1 to simulate
μ2_values = [3, 4]  # Service rates for server 2 to simulate
T = 2000            # Total simulation time
q = 1000            # Initial queue length for L1

plt.figure(figsize=(15, 15))

# Iterate over all combinations of λ, μ1, and μ2
for i, λ in enumerate(λ_values):
    for j, μ1 in enumerate(μ1_values):
        for k, μ2 in enumerate(μ2_values):
            # Perform the simulation with the current parameters
            L1_times, L2_times = simulate_queue(λ, μ1, μ2, T, q)

            # Plot the results in a subplot
            plt.subplot(4, 2, i * 4 + j * 2 + k + 1)
            title = f'λ={λ}, μ1={μ1}, μ2={μ2}'
            plot_queues(L1_times, L2_times, title)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()