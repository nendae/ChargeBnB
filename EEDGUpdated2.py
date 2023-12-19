import numpy as np
import matplotlib.pyplot as plt

#-------------------------------- Variable Input Begin ----------------------------------------------------

# ---------- Payoff Parameters Start ---------------------
V = 40  # Maximum value for a full charge ($)
E = 20   # Energy required to fully charge the vehicle (kWh)

'''
# Utility tariff array if the rates were different based on TOU and seasonality
C_tariff = [0,1]
C_tou_sum = [.23,.23,.23,.23,.23,.23,.23,.23,.23,.23,.30,.30,.30,.30,.30,.30,.30,.30,.30,.30,.30,.23,.23]
C_tou_win = [.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.28,.28,.28,.28,.28,.28,.28,.28,.28,.28,.28,.20,.20]
C_peak_sum = [.23,.23,.23,.23,.23,.23,.23,.23,.23,.23,.30,.30,.30,.30,.30,.30,.30,.30,.30,.30,.30,.23,.23]
C_peak_win = [.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.28,.28,.28,.28,.28,.28,.28,.28,.28,.28,.28,.20,.20]

# Deterime tariff rate
# create if statement based on date/time to determine rate used after getting data from Elibeth
'''
# Fixed price utility tariff rate based on average utility bills
C = 0.45

# ---------- Payoff Parameters End ---------------------------------

#----------- Escort Function Parameters Start ----------------------
# Sensitivity parameters
alpha = [0,.01, .02, .03, .04, .05, .06, .10, .12, .14,.16,.18]  # Time sensitivity parameter for EV owners --surveys
beta = 0.7   # Rating sensitivity parameter for EV owners --surveys
gamma = 0.3  # Price sensitivity parameter for EV charger owners --surveys (TBD)

# Other parameters
time = 10           # Time to drive to charging station (minutes) --provided in app (generic number used for simulation)
rating = 4            # Rating of the charging station --provided in app (generic number used for simulation)
R_avg = 5        # Average rating --provided in app (generic number used for simulation)
p_hat = .32      # Average market price of charging ($/kWh) --data from EV charging apps

#----------- Escort Function Parameters End ----------------------

#----------- State Vectors for supply and demand Start ------------------------------
# Range of prices for supplying and charging
price_range = np.arange(C, V/E, .01)    # From C to V/E
proportion_init = 1/price_range.size    # initial proportion of strategy (price) across population for the state vector
#print("price range: ", price_range)

# Demand (charging) state vector
p_demand = [(p, proportion_init) for p in price_range]
#print("initial proportion array for demand prices: ", p_demand)
#global_p_demand = np.empty((0,2))     # Global array that holds all the generations of state vectors
global_p_demand = []
# Supply (charging) state vector

p_supply = [(p, proportion_init) for p in price_range]
#print("initial proportion array for supply prices: ", p_supply)
#global_p_supply = np.empty((0,2))     # Global array that holds all the generations of state vectors
global_p_supply = []

#----------- State Vectors for supply and demand End -------------------------------------------------


#-------------------------------- Variable Input End ----------------------------------------------------

#-------------------------------- Functions Created Start ----------------------------------------------

# Payoff functions

def payoff(type, state):
    if type == "d": 
        fit = (V - state[0] * E) * state[1]                     # Demand payoff function fk(xk)    
    if type == "s":
        fit =((state[0]  - C) * E) * state[1]                    # Supply payoff function gk(xk)    
    if fit < 0:
        return (state[0], 0) 
    return  (state[0], fit)  
                                             
# Escort functions 

def escortD(state, index, pop_size, alpha, beta, time, rating, R_avg, p_hat):
    # Still in progress
    #loc = (index * 2)/ pop_size   # Maximum value can be 1 for the upper half of the population
    #if loc <= 1:
        #return  alpha[time] * loc + 1    # Price adjustment 
    
    return 1

def escortS(state, index, pop_size, alpha, beta, time, rating, R_avg, p_hat):
    # Currently no function for supply
    return 1
 
# Escort distribution

def escDist(escort_adj, state):     #phi_k(x_k)
    esc_dist = escort_adj * state[1]
    return esc_dist

# Calculate the generation of a population

def stateChange(state, fk,phik,esc_exp, escort_dist):
    total_fit = sum(fit[1] for fit  in fk)          # Total fit calc => fk(X) the summation of the payoff across the entire state vector
    total_escort_dist = sum(escd[1] for escd in escort_dist)     # Total value of the escort function => phik(X) the denominator of WA calc
    total_escort = sum(esc[1] for esc  in phik)     # Total value of the escort function => phik(X) the denominator of WA calc

    # must determine correct calculation for the weighted average numerator (total escort expectation):
    #total_expectation = sum(np.array([(strategy, proportion * total_fit) for strategy, proportion in phik])) # Sum phik(xk)fk(X) sum each escort by total fit
    total_expectation = sum(exp[1] for exp  in esc_exp)     # Sum phik(xk)fk(xk) sum each escort by each fit 

    WA_norm = total_expectation/ total_escort # Normalized escort weighted average of payoffs 
    avg_esc_exp = [(x[0], x[1] * WA_norm) for x in phik]    # phik * f_bar(phi(x))

    #print("avg_esc_exp: ", avg_esc_exp)

    #avg_esc_exp = phik * WA_norm            # phik * f_bar(phi(x))
    delta_state = [ee[1] - aee[1] for ee, aee in zip(esc_exp, avg_esc_exp)]     # Rate of change of each state in state vector after evolution
    #print("delta_state: ", delta_state)
    state       = [(st[0], st[1] + ds) for st, ds in zip(state, delta_state)]  
    #print("state: ", state)
    #print("state: ", state)
    return state # New state vector after evolution

#-------------------------------- Functions Created End ----------------------------------------------

#-------------------------------- Calculate Evolution Start ----------------------------------------------

# Demand strategy evolution function

def evolutionD(p_demand):
    # Initialize demand arrays
    pop_size = len(p_demand)
    fk_demand = np.empty(pop_size, dtype=('f,f'))
    phik_demand = np.empty(pop_size, dtype=('f,f'))
    escort_dist = np.empty(pop_size, dtype=('f,f'))
    escort_expectation_d  = np.empty(pop_size, dtype=('f,f'))
    global global_p_demand

     # Loop through state vector for demand (EV owners)
    for index, state in enumerate(p_demand):
        fk_demand[index] = payoff("d", state)     # Create array for payoff values for each strategy: f(x) = [fk(xk)(xk)]
        escort_adj = escortD(state, index, pop_size, alpha, beta, time, rating, R_avg, p_hat)    # Calculate escort adjustment for each strategy
        #print("escort_adj: ", escort_adj)
        phik_demand[index] = (state[0], escDist(escort_adj, state))  # Create array for escort values for each strategy phik(X)
        #print("phik_demand[", index, "]: ", phik_demand[index])
        escort_dist[index] = (state[0], phik_demand[index][1] * p_demand[index][1]) # Create array for escort distribution: phik(xk)*(xk)
        escort_expectation_d[index] = (state[0], phik_demand[index][1] * fk_demand[index][1]) # Create array for escort distribution: phik(xk)*fk(xk)
        #print("escort_expectation_d[", index, "]: ", escort_expectation_d[index], "      phik_demand[index][1]: ", phik_demand[index][1], "     fk_demand[index][1]: ", fk_demand[index][1])

    #global_p_demand = np.append(global_p_demand, p_demand, axis = 0)       # Update global array with old state vector
    global_p_demand.append(p_demand)
    p_demand = stateChange(p_demand,fk_demand, phik_demand,escort_expectation_d, escort_dist) # New demand state vector
   

    return p_demand

# Supply strategy evolution function

def evolutionS(p_supply):
     # Initialize supply arrays
    pop_size = len(p_supply)
    fk_supply = np.empty(pop_size, dtype=('f,f'))
    phik_supply = np.empty(pop_size, dtype=('f,f'))
    escort_dist = np.empty(pop_size, dtype=('f,f'))
    escort_expectation_s  = np.empty(pop_size, dtype=('f,f'))
    global global_p_supply

    # Loop through state vector for demand (EV charger owners)
    for index, state in enumerate(p_supply):
        fk_supply[index] = payoff("s", state)       # Create array for payoff values for each strategy fk(X)
        escort_adj = escortS(state, index, pop_size, alpha, beta, time, rating, R_avg, p_hat)    # Calculate escort adjustment for each strategy
        phik_supply[index] = (state[0], escDist(escort_adj, state))   # Create array for escort values for each strategy phik(X)
        escort_dist[index] = (state[0], phik_supply[index][1] * p_supply[index][1]) # Create array for escort distribution: phik(xk)*(xk)
        escort_expectation_s[index] = (state[0], phik_supply[index][1] * fk_supply[index][1]) # Create array for escort distribution: phik(xk)*fk(xk)

    #global_p_supply = np.append(global_p_supply, p_supply, axis = 0)       # Update global array with old state vector
    global_p_supply.append(p_supply)
    #print("global supply array: ",global_p_supply )
    p_supply = stateChange(p_supply,fk_supply, phik_supply,escort_expectation_s, escort_dist) # New supply state vector
    return p_supply

#-------------------------------- Calculate Evolution End ----------------------------------------------


#-------------------------------- Calculate Generation Start -------------------------------------------

pct_change_s = 1
pct_change_d = 1
p_demand_last = []
p_supply_last = []
'''
while pct_change >= .01:
    count = 1
    p_demand = evolutionD(p_demand)
    #print("p_demand(", count,"):", p_demand)
    p_supply = evolutionS(p_supply)
    #print("p_supply(", count,"):", p_supply)
    p_demand_last = (global_p_demand[-1])
    p_supply_last = (global_p_supply[-1])    
    max_pc_d = np.max([pdl[1] / pd[1] for pdl, pd in zip(p_demand_last, p_demand)]) 
    max_pc_s = np.max([psl[1] / ps[1] for psl, ps in zip(p_supply_last, p_supply)])
    pct_change = max(max_pc_d, max_pc_s)
    total_pop_prop_d = sum(prop[1] for prop  in p_demand)
    total_pop_prop_s = sum(prop[1] for prop  in p_supply)
    #print("total_pop_prop demand: ", total_pop_prop_d, "        total_pop_prop supply: ", total_pop_prop_s)
    count = count + 1
'''
count = 1
while pct_change_d >= .01:
    p_demand = evolutionD(p_demand)
    for x_d in p_demand:
        if x_d[1] < -1:
            pct_change_d = -1
            break
    if pct_change_d == -1:
        break
    p_demand_last = (global_p_demand[-1])
    max_pc_d = np.max([pdl[1] / pd[1] for pdl, pd in zip(p_demand_last, p_demand)]) 
    pct_change_d = max_pc_d
    print("max_pc_d: ", max_pc_d)
    total_pop_prop_d = sum(prop[1] for prop  in p_demand)
    #print("total_pop_prop demand: ", total_pop_prop_d, "        total_pop_prop supply: ", total_pop_prop_s)
    count = count + 1
    print("count D: ", count)

count = 1
while pct_change_s >= .01:
    p_supply = evolutionS(p_supply)
    for x_s in p_supply:
        if x_s[1] < -1:
            pct_change_s = -1
            break
    if pct_change_s == -1:
        break
    p_supply_last = (global_p_supply[-1])
    max_pc_s = np.max([psl[1] / ps[1] for psl, ps in zip(p_supply_last, p_supply)]) 
    pct_change_s = max_pc_s
    print("max_pc_s: ", max_pc_s)
    total_pop_prop_s = sum(prop[1] for prop  in p_supply)
    #print("total_pop_prop demand: ", total_pop_prop_d, "        total_pop_prop supply: ", total_pop_prop_s)
    count = count + 1
    print("count S: ", count)
#-------------------------------- Calculate Generation End -------------------------------------------


#-------------------------------- Price Comparison Start -------------------------------------------

demand_prices = [p_d for p_d in p_demand_last if p_d[1] > 1.1*(1/price_range.size)]
supply_prices = [p_s for p_s in p_supply_last if p_s[1] > 1.1*(1/price_range.size)]

print("p_demand_last: ", p_demand_last)
print("p_supply_last: ", p_supply_last)

print("demand_prices: ", demand_prices, "    lenth: ", len(demand_prices))
print("supply_prices: ", supply_prices, "    lenth: ", len(supply_prices))




# Extract unique price values
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