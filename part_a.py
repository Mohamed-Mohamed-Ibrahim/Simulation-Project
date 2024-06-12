import numpy as np

def simulate_tandem_queue(T, λ, μ1, μ2, N):
    # Store the average number of customers in the system for each simulation
    avg_customers = []
    
    for _ in range(N):
        # Initialize system state
        t = 0
        L1 = 0
        L2 = 0
        total_customers = 0
        
        while t < T:
            if L1 == 0:
                next_arrival = np.random.exponential(1/λ)
                next_service1 = float('inf')
            else:
                next_arrival = np.random.exponential(1/λ)
                next_service1 = np.random.exponential(1/μ1)
            
            if L2 == 0:
                next_service2 = float('inf')
            else:
                next_service2 = np.random.exponential(1/μ2)
            
            # Determine next event
            next_event = min(next_arrival, next_service1, next_service2)
            t += next_event
            
            if next_event == next_arrival:
                L1 += 1
            elif next_event == next_service1:
                L1 -= 1
                L2 += 1
            elif next_event == next_service2:
                L2 -= 1
            
            # Accumulate the number of customers
            total_customers += (L1 + L2) * next_event
        
        # Average number of customers in the system
        avg_customers.append(total_customers / T)
    
    return np.ceil(np.mean(avg_customers))

def theoretical_avg_customers(λ, μ1, μ2):
    ρ1 = λ / μ1
    ρ2 = λ / μ2
    if ρ1 >= 1 or ρ2 >= 1:
        return float('inf')  # System would be unstable
    return np.ceil((ρ1*ρ1 / (1 - ρ1))) + np.ceil((ρ2*ρ2 / (1 - ρ2)))

# Parameters
λ_values = [1, 5]
μ1_values = [2, 4]
μ2_values = [3, 4]
# T_values = [10, 50, 100, 1000]
T_values = [10, 50, 100]
N = 1000

results = []

for λ in λ_values:
    for μ1 in μ1_values:
        for μ2 in μ2_values:
            for T in T_values:
                sim_avg_customers = simulate_tandem_queue(T, λ, μ1, μ2, N)
                theo_avg_customers = theoretical_avg_customers(λ, μ1, μ2)
                results.append((λ, μ1, μ2, T, sim_avg_customers, theo_avg_customers))

# Print the results
for result in results:
    print(f"λ={result[0]}, μ1={result[1]}, μ2={result[2]}, T={result[3]},\t\t Simulated Avg={np.ceil(result[4])},\t\t Theoretical Avg={np.ceil(result[5])}")
