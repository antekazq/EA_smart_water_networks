#Necessary libraries
import wntr #Water Network Tool for Resilience
import matplotlib.pyplot as plt #For plotting purposes
import networkx as nx #For network visualization

#Load the network model L-Town from an EPANET input file
inp_file = "C:/Users/anton/OneDrive/Documents/EPANET Projects/EPANET_scripts/L-TOWN (1).inp" 
wn = wntr.network.WaterNetworkModel(inp_file)

#Define simulation options
#Set simulation duration to 50 hours (converted to seconds)
sim_duration = 50 * 3600  
wn.options.time.duration = sim_duration
#Hydraulic time step of 1 hour, suitable for scenarios where the hydraulic conditions are not expected to change very rapidly
wn.options.time.hydraulic_timestep = 100  
#Set pattern timestep equal to hydraulic timestep
wn.options.time.pattern_timestep = 100    
wn.options.quality.parameter = "Contamination"

#At 0 hours, the contamination is 10 mg/L
#At 2 hours, the contamination is still 10 mg/L
#Just after 2 hours, the contamination drops to 0 mg/L
#For the remainder of the simulation, the contamination remains at 0 mg/L
pattern_time = [0, 2*3600, 2*3600+1, sim_duration]
pattern_values = [10, 10, 0, 0]  
contamination_pattern = wntr.network.elements.Pattern('contamination_pattern', pattern_time, pattern_values)
wn.add_pattern('contamination_pattern', contamination_pattern)

#Add contamination sources at Reservoirs R1 or R1 and R2, depending on
#if we are simulating 1 or 2 contamination sources
wn.add_source('contamination_R1', 'R1', "CONCEN", 10.0, pattern='contamination_pattern')
wn.add_source('contamination_R2', 'R2', "CONCEN", 10.0, pattern='contamination_pattern')

#Initialize and run the EPANET simulation
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

#Visualization using NetworkX
#Create a graph from the water network model
G = wn.get_graph()  
#Get positions of nodes based on their coordinates
pos = {node_name: (node.coordinates[0], node.coordinates[1]) for node_name, node in wn.nodes()}

#Defining the times for visualization (2 hours after start and at the end of the simulation)
times = [2*3600, sim_duration]  #2 hours in seconds and 50 hours in seconds

#Plot the water quality results at specified times
for time in times:
    #Get quality results at the specified time
    node_quality = results.node['quality'].loc[time]
    #Set node colors based on contamination level (default to 0 if no data)
    node_colors = [node_quality[node] if node in node_quality else 0 for node in G.nodes()]  # Default to 0 if no data

    plt.figure(figsize=(12, 8))
    #Draw the network graph with node colors representing contamination levels
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=10, cmap=plt.cm.jet)
    #Add a color bar to represent contaminant concentration
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet), ax=plt.gca(), orientation='vertical', label='Contaminant Concentration (mg/ml)')
    #Add labels and title to the plot
    plt.ylabel(time)
    plt.title(f'Water Network Contamination Spread After {time/3600} Hours')
    plt.show()

#Save the simulation results to a file using pickle
import pickle

#Save the results object to a file
with open('simulation_results.pkl', 'wb') as f:
    pickle.dump(results, f)
