#Necessary libraries
import wntr #Water Network Tool for Resilience
import matplotlib.pyplot as plt #For plotting purposes
import networkx as nx #For network visualization
import matplotlib.gridspec as gridspec #For creating axes

#NEW SIMULATION FOR CASE STUDY 2

#Load the network model
inp_file = "C:/Users/anton/OneDrive/Documents/EPANET Projects/EPANET_scripts/L-TOWN (1).inp" 
wn = wntr.network.WaterNetworkModel(inp_file)

#Define simulation options
#Set simulation duration to 24 hours (converted to seconds)
sim_duration = 24 * 3600  
wn.options.time.duration = sim_duration
#Hydraulic time step of 1 hour, suitable for scenarios where the hydraulic conditions are not expected to change very rapidly
wn.options.time.hydraulic_timestep = 100  
#Set pattern timestep equal to hydraulic timestep
wn.options.time.pattern_timestep = 100    
wn.options.quality.parameter = "Contamination"

#At 0 hours, the contamination is 10 mg/L
#At 10 hours, the contamination is still 10 mg/L
#Just after 10 hours, the contamination drops to 0 mg/L
#For the remainder of the simulation, the contamination remains at 0 mg/L
pattern_time = [0, 10*3600, 10*3600+1, sim_duration] 
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

#Get contamination results at the last time step
node_contamination = results.node['quality'].iloc[-1]

#Normalize the node_quality values for color mapping, Hence 0-1
max_quality = node_contamination.max()
min_quality = node_contamination.min()
norm_node_quality = (node_contamination - min_quality) / (max_quality - min_quality)

#Create network graph
G = wn.get_graph()
#Get positions of nodes based on their coordinates
pos = {node_name: (node.coordinates[0], node.coordinates[1]) for node_name, node in wn.nodes()}

#Plot setup
fig = plt.figure(figsize=(12, 8))
plt.axis("off")
plt.title('Case Study Two: 24-Hour Contamination')
gs = gridspec.GridSpec(8, 12)  #Split into 12 columns

#Assign axes. Works like different "blank papers"
ax_network = fig.add_subplot(gs[0:8, 0:10])  #Graph
ax_cbar = fig.add_subplot(gs[1:7, 11:12])    #Color bar

#Drawing the network graph with contamination levels as node colors
nx.draw(G, pos, node_color=node_contamination, with_labels=False, cmap=plt.cm.jet, ax=ax_network, node_size=10)

#Colorbar setup
norm = plt.Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm),
    cax=ax_cbar, orientation='vertical', label='Contamination Level (mg/L)'
)
cbar.ax.set_aspect(20)  #Adjust the aspect ratio
cbar.ax.set_frame_on(True)  #Add a frame to the color bar

for spine in ax_network.spines.values():
    spine.set_visible(False)

#Display
plt.show()

import pickle

#Save the results object to a file
with open('case_study_two_simulation.pkl', 'wb') as f:
    pickle.dump(results, f)