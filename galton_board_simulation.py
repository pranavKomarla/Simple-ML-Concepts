import numpy as np
import matplotlib.pyplot as plt

def simulate_galton_board(M, num_simulations=1000):
    results = np.random.binomial(M, 0.5, num_simulations)  # Simulate ball drops
    unique, counts = np.unique(results, return_counts=True)
    pmf = counts / num_simulations  # Compute PMF
    
    plt.bar(unique, pmf, alpha=0.6, label=f'M = {M}')
    plt.xlabel('Final Position (Number of Right Moves)')
    plt.ylabel('Probability Mass Function (PMF)')
    plt.title(f'Galton Board Simulation (M={M})')
    plt.legend()

plt.figure(figsize=(10, 6))
for M in [5, 10, 100]:
    simulate_galton_board(M)

plt.show()
plt.savefig('galton_board_simulation.png') 

