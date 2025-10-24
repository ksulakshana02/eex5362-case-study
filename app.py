import simpy
import random
import numpy as np
import matplotlib.pyplot as plt


SIM_TIME = 120  # minutes of peak time
MEAN_INTERARRIVAL = 1.5  # average time between customers arriving
MIN_SERVICE_REG = 2  # min service for regular baskets
MAX_SERVICE_REG = 7  # max for regular
MIN_SERVICE_EXP = 1  # quick for express
MAX_SERVICE_EXP = 3
PROB_SMALL_BASKET = 0.4  # about 40% of folks with 10 items or less
NUM_REPS = 5  # a few runs to smooth out the randomness

class SupermarketCheckoutSim:
    def __init__(self, env, num_regular_counters, has_express_lane=False):
        self.env = env
        self.regular_counters = simpy.Resource(env, capacity=num_regular_counters)
        self.express_lane = simpy.Resource(env, capacity=1) if has_express_lane else None
        self.wait_times = []
        self.served_count = 0
        self.total_service_reg = 0.0
        self.total_service_exp = 0.0
        self.max_queue_length = 0
        self.queue_history = []

    def customer_arrival(self):
        # show up, choose line, wait, pay, leave.
        arrival_time = self.env.now
        is_small_basket = random.random() < PROB_SMALL_BASKET
        
        if self.express_lane and is_small_basket:
            # express lane
            resource = self.express_lane
            service_time = random.uniform(MIN_SERVICE_EXP, MAX_SERVICE_EXP)
        else:
            # Regular line
            resource = self.regular_counters
            service_time = random.uniform(MIN_SERVICE_REG, MAX_SERVICE_REG)
        
        with resource.request() as request:
            yield request
            wait_time = self.env.now - arrival_time
            self.wait_times.append(wait_time)
            
            # Check current queue load
            reg_load = len(self.regular_counters.queue) + self.regular_counters.count
            exp_load = len(self.express_lane.queue) + self.express_lane.count if self.express_lane else 0
            current_total = reg_load + exp_load
            self.max_queue_length = max(self.max_queue_length, current_total)
            
            # checkout
            yield self.env.timeout(service_time)
            self.served_count += 1
            if self.express_lane and is_small_basket:
                self.total_service_exp += service_time
            else:
                self.total_service_reg += service_time

    def queue_monitor(self):
        while True:
            yield self.env.timeout(1)  # Check every minute
            reg_load = len(self.regular_counters.queue) + self.regular_counters.count
            exp_load = len(self.express_lane.queue) + self.express_lane.count if self.express_lane else 0
            total_in_system = reg_load + exp_load
            self.queue_history.append((self.env.now, total_in_system))

def run_simulation(num_regular, has_express=False, num_reps=NUM_REPS):
    all_wait_times = []
    all_max_queues = []
    all_served = []
    all_util_reg = []
    all_util_exp = []
    queue_histories = []
    
    for rep in range(num_reps):
        env = simpy.Environment()
        sim = SupermarketCheckoutSim(env, num_regular, has_express)
        
        # monitors and arrivals
        env.process(sim.queue_monitor())
        def arrival_process():
            while True:
                yield env.timeout(random.expovariate(1 / MEAN_INTERARRIVAL))  # Poisson arrivals
                env.process(sim.customer_arrival())
        env.process(arrival_process())
        
        env.run(until=SIM_TIME)
        
        # Collect data
        if sim.wait_times:
            all_wait_times.extend(sim.wait_times)
        all_max_queues.append(sim.max_queue_length)
        all_served.append(sim.served_count)
        util_reg = (sim.total_service_reg / (num_regular * SIM_TIME)) * 100 if num_regular > 0 else 0
        all_util_reg.append(util_reg)
        if has_express:
            util_exp = (sim.total_service_exp / SIM_TIME) * 100
            all_util_exp.append(util_exp)
        
        queue_histories.append(sim.queue_history)

    avg_wait = np.mean(all_wait_times) if all_wait_times else 0
    avg_max_q = np.mean(all_max_queues)
    avg_throughput = (np.mean(all_served) / SIM_TIME) * 60
    avg_util_reg = np.mean(all_util_reg)
    avg_util_exp = np.mean(all_util_exp) if has_express and all_util_exp else None
    
    results = {
        'avg_wait': avg_wait,
        'max_queue': avg_max_q,
        'throughput': avg_throughput,
        'util_reg': avg_util_reg,
        'util_exp': avg_util_exp
    }
    
    return results, queue_histories[-1]  # Use last history for plot

if __name__ == "__main__":
    print("------ Simulating Checkout Queues at Keells/Cargills ------")
    print(f"Setup: {SIM_TIME} min sim, arrivals every {MEAN_INTERARRIVAL} min on avg.")
    print(f"Service: Regular {MIN_SERVICE_REG}-{MAX_SERVICE_REG} min, Express {MIN_SERVICE_EXP}-{MAX_SERVICE_EXP} min.")
    print(f"{PROB_SMALL_BASKET*100}% chance of small basket for express lane.\n")
    
    # Scenario 1: Baseline - 4 counters
    print("------ Scenario 1: 4 Regular Counters ------")
    res1, hist1 = run_simulation(4, False)
    print(f"Average Wait Time: {res1['avg_wait']:.2f} minutes")
    print(f"Max Queue Length: {res1['max_queue']:.1f} customers")
    print(f"Throughput: {res1['throughput']:.1f} customers per hour")
    print(f"Cashier Utilization: {res1['util_reg']:.1f}%")
    print()
    
    # Scenario 2: 6 counters
    print("------ Scenario 2: 6 Regular Counters ------")
    res2, hist2 = run_simulation(6, False)
    print(f"Average Wait Time: {res2['avg_wait']:.2f} minutes")
    print(f"Max Queue Length: {res2['max_queue']:.1f} customers")
    print(f"Throughput: {res2['throughput']:.1f} customers per hour")
    print(f"Cashier Utilization: {res2['util_reg']:.1f}%")
    print()
    
    # Scenario 3: Express lane (5 regular + 1 express)
    print("------ Scenario 3: 5 Regular + 1 Express Lane ------")
    res3, hist3 = run_simulation(5, True)
    print(f"Average Wait Time: {res3['avg_wait']:.2f} minutes (overall)")
    print(f"Max Queue Length: {res3['max_queue']:.1f} customers")
    print(f"Throughput: {res3['throughput']:.1f} customers per hour")
    print(f"Regular Utilization: {res3['util_reg']:.1f}%")
    print(f"Express Utilization: {res3['util_exp']:.1f}%")
    print()
    
    
    # Bar chart
    scenarios = ['4 Counters', '6 Counters', '5+1 Express']
    wait_times = [res1['avg_wait'], res2['avg_wait'], res3['avg_wait']]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(scenarios, wait_times, color=['red', 'orange', 'green'])
    plt.title('Average Customer Wait Times by Scenario')
    plt.ylabel('Wait Time (minutes)')
    plt.xticks(rotation=15)
    # Add value labels on bars
    for bar, wait in zip(bars, wait_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{wait:.1f}', 
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('wait_times.png', dpi=300, bbox_inches='tight')
    
    # Bar chart: Utilization
    utils_reg = [res1['util_reg'], res2['util_reg'], res3['util_reg']]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(scenarios, utils_reg, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Regular Cashier Utilization by Scenario')
    plt.ylabel('Utilization (%)')
    plt.ylim(0, 100)
    for bar, util in zip(bars, utils_reg):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{util:.0f}%', 
                 ha='center', va='bottom')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('utilization.png', dpi=300, bbox_inches='tight')
    
    # Line plot: Queue evolution over time
    if hist1:
        times, queue_lens = zip(*hist1)
        plt.figure(figsize=(10, 6))
        plt.plot(times, queue_lens, linewidth=2, color='purple')
        plt.title('Queue Length Evolution Over Time (Scenario 1: 4 Counters)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Customers in System (Queue + Being Served)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('queue_evolution.png', dpi=300, bbox_inches='tight')
    
    print("All done!")