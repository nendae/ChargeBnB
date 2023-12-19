import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#-------------------------------- Variable Input Begin ----------------------------------------------------

# ---------- Payoff Parameters  ---------------------
V = 42          # Value for a full charge using average from survey answers ($)
E = 30.8        # Energy required to fully charge the vehicle (kWh) Tesla Mondel Y (75kWh battery size) at 41%
C = 0.46      # Average utility bill rate provided
p_hat = 1.315      # Average market price of charging ($/kWh) --data from EV charging apps

#----------- Escort Function Parameters  ----------------------

time = 10           # Time to drive to charging station (minutes) --provided in app (generic number used for simulation)
rating = 4.82            # Rating of the charging station --provided in app (generic number used for simulation)
R_avg = 4.7        # Average rating --provided in app (generic number used for simulation)


#----------- State Vectors for supply and demand Start ----------------------------

# Range of prices for supplying and charging
p_max = min(p_hat *1.2, V/E)
p_min = max(C, p_hat*.7)
price_range = np.arange(p_min, p_max, .01)    # From C to max of avg market prices of nearby chargers or average surveyed utility V/E 
proportion_init = 1/price_range.size    # initial proportion of strategy (price) across population for the state vector
print("p_max: ", p_max)
print("p_min: ", p_min)
print("price_range.size: ", price_range.size)
print("proportion_init: ", proportion_init)

# Demand (charging) state vector
p_demand = np.array([(p, proportion_init) for p in price_range])
p_supply = [(p, proportion_init) for p in price_range]
global_p_demand = []
global_p_supply = []


#-------------------------------- Variable Input End --------------------------------------------------


#-------------------------------- Functions Created Start ---------------------------------------------

# Payoff function

def payoff(type, state_payoff):
    global V
    if type == "d": 
        return (state_payoff[0], ((V - state_payoff[0] * E)) * state_payoff[1])       # Demand payoff function fk(xk)
    if type == "s":
        return (state_payoff[0], ((state_payoff[0]  - C) * E) * state_payoff[1])     # Supply payoff function gk(xk)
    return 0                                                    # Incorrect type passed in

# Escort functions 
def escortD(time, state_escortd):
    
    #Escort function for time to charging stations. Returns the adjustment factor to apply to the state proportion.
    def timeEsc(time,state_escortd):
        # This example uses time t = 10 => y= 0.1799ln(x) + 0.6171
        top_disc = p_hat - (p_hat*.2)
        time_esc_fn = {                 # Create a dictionary based on different times to store the various function variables 
            1: (-0.085,-0.1573),
            2: (-0.021, 0.0211),
            3: (-0.025, -0.0157),
            10: (0.1779, 0.6171)
        }

        sorted_time_fn = sorted(time_esc_fn.keys())
        for time_key in reversed(sorted_time_fn):
            if time_key <= time:
                time = time_key
                break
        
        
        average_diff = (p_max + state_escortd[0])/2        
        #discount = (top_disc - state_escortd[0] )/average_diff - 1
        #discount = top_disc - abs(state_escortd[0] - top_disc)

        discount = (p_max - state_escortd[0])/average_diff
        m, b = time_esc_fn[time]
        time_adj = m * math.log(discount) + b
        return time_adj

    esc_time = timeEsc(time,state_escortd)
    if esc_time > 0:
        esc_time = esc_time
    else:
        esc_time = 0
    

    
    return esc_time


def escortS(state_escS):
    if state_escS[0] >= p_hat:
        return rating/R_avg
    return 1

# Escort distribution
def escDist(escort_adj, state_escD):     #phi_k(x_k)
    return escort_adj * state_escD[1]

max_pc = 0
# Calculate the generation of a population
def stateChange(state_sc, fk,phik,esc_exp):
    global max_pc
    total_escort = sum(esc[1] for esc  in phik)     # Total value of the escort function => phik * (X) the denominator of WA calc

    # must determine correct calculation for the weighted average numerator (total escort expectation):
    #total_expectation = sum(np.array([(strategy, proportion * total_fit) for strategy, proportion in phik])) # Sum phik(xk)fk(X) sum each escort by total fit
    
    total_expectation = sum(exp[1] for exp  in esc_exp)     # Sum phik(xk)fk(xk) sum each escort by each fit 
    WA_norm = total_expectation/ total_escort # Normalized escort weighted average of payoffs
    avg_esc_exp = [(x[0], x[1] * WA_norm) for x in phik]    # phik * f_bar(phi(x))

    #avg_esc_exp = phik * WA_norm            # phik * f_bar(phi(x))
    delta_state = [ee[1] - aee[1] for ee, aee in zip(esc_exp, avg_esc_exp)]     # Rate of change of each state in state vector after evolution
    state_sc       = [(st[0], st[1] + ds) for st, ds in zip(state_sc, delta_state)] 
    # Assuming your demand array looks like this
    updated_state_sc = [(price, percent if percent >= 0 else 0) for price, percent in state_sc]
    total_state_proportions = sum(tsp[1] for tsp  in updated_state_sc) 
    norm_state_sc = [(x[0], x[1] / total_state_proportions) for x in updated_state_sc] 
    max_pc = np.max([ds / uss[1] if uss[1] != 0 else 0 for ds, uss in zip(delta_state, updated_state_sc)]) 
    return norm_state_sc # New state vector after evolution

#-------------------------------- Functions Created End --------------------------------------------------

#-------------------------------- Calculate Evolution Start ----------------------------------------------

# Demand strategy evolution function
count = 1
def evolutionD(p_demand_evo):
    global count
    # Initialize demand arrays
    fk_demand = np.empty(len(p_demand_evo), dtype=('f,f'))
    phik_demand = np.empty(len(p_demand_evo), dtype=('f,f'))
    state_vector = np.empty(len(p_demand_evo), dtype=('f,f'))
    escort_expectation_d  = np.empty(len(p_demand_evo), dtype=('f,f'))
    phik_demand_nox = np.empty(len(p_demand_evo), dtype=('f,f'))

    global global_p_demand
   
     # Loop through state vector for demand (EV owners)
    for index, state_evo in enumerate(p_demand_evo):
        fk_demand[index] = payoff("d", state_evo)     # Create array for payoff values for each strategy fk(X)
        state_vector[index] = (state_evo[0],state_evo[1])
        escort_adj = escortD(time, state_evo)    # Calculate escort adjustment for each strategy
        phik_demand_nox[index] = (state_evo[0], escort_adj)     # Create array for escort values for each strategy phik 
        phik_demand[index] = (state_evo[0], escDist(escort_adj, state_evo))  # Create array for escort values for each strategy phik * (x_k)
        #escort_expectation_d[index] = (state_evo[0], phik_demand[index][1] * fk_demand[index][1]) # Create array for escort distribution: phik * (xk) * fk * (xk)
        escort_expectation_d[index] = (state_evo[0], phik_demand_nox[index][1] * fk_demand[index][1]) # Create array for escort distribution: phik * (xk) * fk(xk)
   
    
    print("count: ", count) 
    print("state_vector: ", state_vector)
    print("p_demand: ", p_demand)
    print("fk_demand: ", fk_demand)
    print("phik_demand_nox: ", phik_demand_nox)
    print("phik_demand: ", phik_demand)
    print("escort_expectation_d: ", escort_expectation_d)
    
    #global_p_demand = np.append(global_p_demand, p_demand, axis = 0)       # Update global array with old state vector
    global_p_demand.append(p_demand_evo)
    p_demand_evo = stateChange(p_demand_evo,fk_demand, phik_demand,escort_expectation_d) # New demand state vector
    count = count + 1

    return p_demand_evo

# Supply strategy evolution function

def evolutionS(p_supply_evo):
     # Initialize supply arrays
    global count
    fk_supply = np.empty(len(p_supply_evo), dtype=('f,f'))
    phik_supply = np.empty(len(p_supply_evo), dtype=('f,f'))
    escort_expectation_s  = np.empty(len(p_supply_evo), dtype=('f,f'))
    phik_supply_nox = np.empty(len(p_supply_evo), dtype=('f,f'))
    global global_p_supply

    # Loop through state vector for demand (EV charger owners)
    for index, state_evo in enumerate(p_supply_evo):
        fk_supply[index] = payoff("s", state_evo)       # Create array for payoff values for each strategy fk(X)
        escort_adj = escortS(state_evo)    # Calculate escort adjustment for each strategy
        phik_supply[index] = (state_evo[0], escDist(escort_adj, state_evo))   # Create array for escort values for each strategy phik(X)
        phik_supply_nox[index] = (state_evo[0], escort_adj)     # Create array for escort values for each strategy phik 
        escort_expectation_s[index] = (state_evo[0], phik_supply_nox[index][1] * fk_supply[index][1]) # Create array for escort distribution: phik(xk)*fk(xk)
   
    global_p_supply.append(p_supply_evo)
    p_supply_evo = stateChange(p_supply_evo,fk_supply, phik_supply,escort_expectation_s) # New supply state vector
    count = count + 1

    return p_supply_evo


#-------------------------------- Calculate Evolution End ----------------------------------------------


#-------------------------------- Calculate Generation Start -------------------------------------------

pct_change = 1
p_demand_last = []
p_demand_last = []
#while pct_change >= .01:
while count < 10:
    p_demand = evolutionD(p_demand)
    max_pc_d = max_pc
    p_supply = evolutionS(p_supply)
    max_pc_s = max_pc
    p_demand_last = (global_p_demand[-1])
    p_supply_last = (global_p_supply[-1])    
    pct_change = max(max_pc_d, max_pc_s)


#-------------------------------- Calculate Generation End -------------------------------------------


#-------------------------------- Price Comparison Start ---------------------------------------------

#demand_prices = [p_d for p_d in p_demand_last if p_d[1] > 1.1*(1/price_range.size)]
#supply_prices = [p_s for p_s in p_supply_last if p_s[1] > 1.1*(1/price_range.size)]

demand_prices = [p_d for p_d in p_demand_last if p_d[1] > 0 ]
supply_prices = [p_s for p_s in p_supply_last if p_s[1] > 0]

print("demand_prices: ", demand_prices, "    lenth: ", len(demand_prices))
print("supply_prices: ", supply_prices, "    lenth: ", len(supply_prices))

# Initialize the columns of the table
price = []
supply = []
supply_growth = []
demand = []
demand_growth = []

# Create a dictionary from array_b for faster lookup
dict_b = dict(demand_prices)

# Iterate over array_a and compare with array_b

for p, s in supply_prices:
    if p in dict_b:
        d = dict_b[p]
        price.append(p)
        supply.append(s)
        supply_growth.append(s / 10)
        demand.append(d)
        demand_growth.append(d / 10)


# Printing the table for visualization (you can format it as needed)
if len(price) == 0:
    print("No price overlap")
else:
    print("Price", "Supply", "Supply Growth", "Demand", "Demand Growth")
    for i in range(len(price)):
        print(price[i], supply[i], supply_growth[i], demand[i], demand_growth[i])



prices = sorted(set([price for price, _ in supply_prices] + [price for price, _ in demand_prices]))

# Normalize the percent values for supply and demand
total_supply = sum(percent for _, percent in supply_prices)
total_demand = sum(percent for _, percent in demand_prices)
normalized_supply = {price: percent / total_supply for price, percent in supply_prices}
normalized_demand = {price: percent / total_demand for price, percent in demand_prices}

# Create the table
table = []
for price in prices:
    supply_pct = normalized_supply.get(price, 0)  # 0 if the price is not in supply_array
    demand_pct = normalized_demand.get(price, 0)  # 0 if the price is not in demand_array
    table.append((price, supply_pct, demand_pct))

# Display the table
print("Price", "Supply_pct", "Demand_pct")
for row in table:
    print(row)






# --------------------------------Create Plots Start ---------------------------------------------------


# Plotting each sublist as a separate line
for index, sublist in enumerate(global_p_supply):
    prices = [item[0] for item in sublist]
    percents = [item[1] for item in sublist]
    plt.plot(prices, percents, marker='o', label=f'Evolution {index +1}')  # Using 'o' to denote points

plt.xlabel('Price (pr)')
plt.ylabel('Pop Proportion')
plt.title('EVSE Strategy and Population Evolution')
plt.legend()
plt.show()

for index, sublist in enumerate(global_p_demand):
    prices = [item[0] for item in sublist]
    percents = [item[1] for item in sublist]
    plt.plot(prices, percents, marker='o', label=f'Evolution {index +1}')  # Using 'o' to denote points

plt.xlabel('Price (pr)')
plt.ylabel('Pop Proportion')
plt.title('EV Strategy and Population Evolution')
plt.legend()
plt.show()


'''
# Create a matrix to store evolutions of supply and demand populations for each strategy (price)
payoff_matrix_ev_owner = np.zeros((len(price_range), len(price_range)))
payoff_matrix_charger_owner = np.zeros((len(price_range), len(price_range)))

# Populate the payoff matrix
for i, pb in enumerate(price_range):
    for j, pc in enumerate(price_range):
        payoff_matrix_ev_owner[i, j] = fk(pb, pc)
        payoff_matrix_charger_owner[i, j] = gk(pb, pc)



# Plot the payoff matrices
plt.figure(figsize=(12, 5))

# Payoff matrix for EV owners
plt.subplot(1, 2, 1)
plt.imshow(payoff_matrix_ev_owner, origin='lower', extent=[C, V/E, C, V/E])
plt.colorbar(label='Payoff')
plt.title('Payoff Matrix for EV Owners')
plt.xlabel('Charging Price ($/kWh)')
plt.ylabel('Bid Price ($/kWh)')

# Payoff matrix for charger owners
plt.subplot(1, 2, 2)
plt.imshow(payoff_matrix_charger_owner, origin='lower', extent=[C, V/E, C, V/E])
plt.colorbar(label='Payoff')
plt.title('Payoff Matrix for Charger Owners')
plt.xlabel('Charging Price ($/kWh)')
plt.ylabel('Bid Price ($/kWh)')

plt.tight_layout()
plt.show()


# --------------------------------Create Plots End ---------------------------------------------------

'''