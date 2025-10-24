# Supermarket Checkout Queue Simulation

This project simulates checkout queues at **Keells** and **Cargills** supermarkets using `SimPy` for discrete-event simulation and `Matplotlib` for data visualization.

The simulation compares three scenarios:
1. **4 Regular Counters** – Typical setup during moderate rush hours.  
2. **6 Regular Counters** – Additional counters to reduce waiting time.  
3. **5 Regular + 1 Express Lane** – Includes an express counter for small baskets.


## Requirements
Make sure you have Python installed, then install dependencies:
```bash
pip install simpy matplotlib numpy
````

## Usage
Run the simulation script in your terminal:
```bash
python app.py
````

## Outputs
- **Terminal Results** : Summary of each scenario’s performance.
- **Saved Charts** : Visual comparison of service quality and efficiency.
