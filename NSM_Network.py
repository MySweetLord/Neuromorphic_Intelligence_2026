import sys
# sys.path.append('.')
sys.path.append('/home/dyuser-e/dynap-se1')
import samna
import numpy as np
import matplotlib.pyplot as plt
import samna.dynapse1 as dyn1
import dynapse1utils as ut
from netgen import Neuron, NetworkGenerator
import params
from params import set_params
import time
from IPython.display import Image

# open DYNAP-SE1 board to get Dynapse1Model

# The open_dynapse1 utility will initialize communication with the device
# - gui=False: do not open any graphical interface --> the DYNAPSE1 GUI does not work if you connect remotely
# - select_device=True: allow user/device selection if multiple boards are detected --> this is necessary when you are connecting via Zemo, since there is more than 1 board attached

model, _ = ut.open_dynapse1(gui=False, select_device=True)

# model = ut.open_specific_device_in_sequence(2)


api = model.get_dynapse1_api()          # get Dynapse1 api from the model
set_params(model)                       # set standard parameters


config1     = model.get_configuration() 
param_group = config1.chips[0].cores[0].parameter_group 

# set AMPA (fast excitatory) weight 
param_group.param_map['PS_WEIGHT_EXC_F_N'].fine_value   = 100
param_group.param_map['PS_WEIGHT_EXC_F_N'].coarse_value = 5

# set NMDA (slow excitatory) weight
param_group.param_map['PS_WEIGHT_EXC_S_N'].fine_value   = 80
param_group.param_map['PS_WEIGHT_EXC_S_N'].coarse_value = 5

# set GABA B (slow inhibitory) weight
param_group.param_map['PS_WEIGHT_INH_S_N'].fine_value   = 80
param_group.param_map['PS_WEIGHT_INH_S_N'].coarse_value = 5

# set GABA A (fast inhibitory) weight
param_group.param_map['PS_WEIGHT_INH_F_N'].fine_value   = 80
param_group.param_map['PS_WEIGHT_INH_F_N'].coarse_value = 5

# set NMDA (slow exc) tau   
# remember the higher the tau param, the higher the leak, the faster the decay
# param_group.param_map['NPDPIE_TAU_S_P'].fine_value   = 50
# param_group.param_map['NPDPIE_TAU_S_P'].coarse_value = 4

# set AMPA (fast exc) tau 
# param_group.param_map['NPDPIE_TAU_F_P'].fine_value   = 50
# param_group.param_map['NPDPIE_TAU_F_P'].coarse_value = 4

# Neuron membrane voltage gain
# param_group.param_map['IF_THR_N'].fine_value = 80 
# param_group.param_map['IF_THR_N'].coarse_value = 4




model.update_parameter_group(param_group, 0, 0)

NE    = 16         # neurons per excitatory state population
NI    = 16         # inhibitory population size
NSum  = 16         # Summation population size
NT    = 16         # Transition population size
NIT   = 7          # Transition Inhibitoring population size
NIIT  = 7          # ransition Inhibitoring Inhibitoring population size

NS = 3             # Nr of States (Controlls Nr of S, T, IT and IIT pops and Nr of Spike generators)


# ------------------------------------------------------------
# WTA 
# ------------------------------------------------------------

p_S_S_base, p_S_S_frac     = 0.6, 0.3       # AMPA base prob, NMDA fraction
p_S_I_base, p_S_I_frac     = 0.6, 0.0       # AMPA base prob, NMDA fraction
p_I_S_base, p_I_S_frac     = 0.3, 0.5       # GABA_A base prob, GABA_B fraction
p_I_I_base, p_I_I_frac     = 0.0, 0.0       # GABA_A base prob, GABA_B fraction

# ------------------------------------------------------------
# SUM POP + TRANSITION POP INHIBITORS
# ------------------------------------------------------------

p_S_Sum_base, p_S_Sum_frac       = 0.3, 0.0     # AMPA base prob, NMDA fraction
p_Sum_Sum_base, p_Sum_Sum_frac   = 0.3, 0.0     # AMPA base prob, NMDA fraction

p_Sum_IT_base, p_Sum_IT_frac     = 0.8, 0.0     # AMPA base prob, NMDA fraction
p_S_IIT_base, p_S_IIT_frac       = 0.5, 0.0     # AMPA base prob, NMDA fraction

p_IT_T_base, p_IT_T_frac         = 0.8, 0.0     # GABA_A base prob, GABA_B fraction
p_IIT_IT_base, p_IIT_IT_frac     = 0.8, 0.0     # GABA_A base prob, GABA_B fraction


# ------------------------------------------------------------
# TRANSITION POP
# ------------------------------------------------------------

p_T_T_base, p_T_T_frac      = 0.6, 0.3       # AMPA base prob, NMDA fraction
p_T_S_base, p_T_S_frac      = 0.3, 0.0       # AMPA base prob, NMDA fraction

# ------------------------------------------------------------
# SPIKE GENERATOR CONNECTION
# ------------------------------------------------------------

p_G_T                       = 0.4  # spike generator to Transition population (deterministic)


# ------------------------------------------------------------
# FIRING RATES
# ------------------------------------------------------------

lr    = 40      # low rate (Hz)
hr    = 150     # high rate (Hz)

stim_rates    = np.linspace(lr,lr,20)
stim_rates_hz = [[rate, rate, rate] for rate in stim_rates ]

stim_rates_hz[10][1] = hr
stim_rates_hz[11][1] = hr
stim_rates_hz[12][1] = hr
stim_rates_hz[13][1] = hr

dt_stim       = 0.5
t_steps_stim  = dt_stim*np.ones_like(stim_rates)
t_post_stim   = 2.0
t_tot         = sum(t_steps_stim) + t_post_stim

print(t_tot)




rng     = np.random.default_rng(seed = 1)
net_gen = NetworkGenerator()
net_gen.clear_network()



# ------------------------------------------------------------
# CONSTRUCT CONNECTION MATRIX
# ------------------------------------------------------------

N_hw = NE*NS + NT*NS + NIT*NS + NIIT*NS + NI + NSum
N_tot = N_hw + NS

C = np.zeros((N_tot, N_tot))

def get_idx(n):
    # real neurons (core 0)
    if n.core_id == 0:
        return n.neuron_id - 1

    # spike generators (core 1)
    elif n.core_id == 1:
        return N_hw + (n.neuron_id - 1)

    else:
        raise ValueError("Unknown core_id")

def add_conn(src_neuron, tgt_neuron, syn_type):
    net_gen.add_connection(src_neuron, tgt_neuron, syn_type)

    src_neuron_idx = get_idx(src_neuron)
    tgt_neuron_idx = get_idx(tgt_neuron)

    if syn_type == dyn1.Dynapse1SynType.AMPA:
        C[src_neuron_idx, tgt_neuron_idx] = 1

    elif syn_type == dyn1.Dynapse1SynType.GABA_A:
        C[src_neuron_idx, tgt_neuron_idx] = -1

    elif syn_type == dyn1.Dynapse1SynType.NMDA:
        C[src_neuron_idx, tgt_neuron_idx] = 2

    elif syn_type == dyn1.Dynapse1SynType.GABA_B:
        C[src_neuron_idx, tgt_neuron_idx] = -2



# Population Making Functions

def make_population(pop_size, next_neuron_id, chip=0, core=0):
    """Creates one population given population size and id for first neuron."""
    pop_ids = [(chip, core, nid) for nid in range(next_neuron_id, next_neuron_id + pop_size)]
    pop     = [Neuron(chip_, core_, nid) for chip_, core_, nid in pop_ids ]

    next_neuron_id += pop_size

    return pop, pop_ids, next_neuron_id


def make_populations(n_pops, pop_size, next_neuron_id, chip=0, core=0):
    """Create multiple populations of same type"""
    pops    = []
    all_ids = []

    for _ in range(n_pops):
        pop, pop_ids, next_neuron_id = make_population(
            pop_size,
            next_neuron_id,
            chip,
            core
        )

        pops.append(pop)
        all_ids.extend(pop_ids)

    return pops, all_ids, next_neuron_id



spikegen_ids = [(0, 1, i + 1) for i in range(NS)]                                                          # chip, core, neuron
spikegens    = [Neuron(chip, core, nid, True) for chip, core, nid in spikegen_ids]


next_neuron_id = 1

S_pops,   S_ids,   next_neuron_id = make_populations(NS, NE,    next_neuron_id)
T_pops,   T_ids,   next_neuron_id = make_populations(NS, NT,    next_neuron_id)
IT_pops,  IT_ids,  next_neuron_id = make_populations(NS, NIT,   next_neuron_id)
IIT_pops, IIT_ids, next_neuron_id = make_populations(NS, NIIT,  next_neuron_id)

I_pop,    I_ids,   next_neuron_id = make_population(NI,    next_neuron_id)
Sum_pop,  Sum_ids, next_neuron_id = make_population(NSum,  next_neuron_id)

assert next_neuron_id < 257, "Too many neurons on one core!"




def connect_pops(source_pop, target_pop, p_base, p_frac, syn_type_fast, syn_type_slow, self_connect = False):

    for i, src_neuron in enumerate(source_pop):
        for j, tgt_neuron in enumerate(target_pop):
            if rng.random() < p_base:

                if source_pop is target_pop:
                    if not self_connect and src_neuron is tgt_neuron:
                        continue   

                if rng.random() < p_frac:
                    add_conn(src_neuron,  tgt_neuron, syn_type_slow)

                else:
                    add_conn(src_neuron,  tgt_neuron, syn_type_fast)





# ------------------------------------------------------------
# CONNECT SPIKEGENS -> TRANSITION POPULATIONS - DETERMINISTIC
# ------------------------------------------------------------
for j in range(NS):
    k = 0
    for src_neuron in T_pops[j]:
        if k < NT * p_G_T:
            add_conn(spikegens[j], src_neuron, dyn1.Dynapse1SynType.AMPA)
            k += 1





for s in range(NS):                                 # STATE POPULATION RECURRENCE: S_i -> S_i
    connect_pops(S_pops[s], S_pops[s], 
                 p_S_S_base, p_S_S_frac, 
                 dyn1.Dynapse1SynType.AMPA, 
                 dyn1.Dynapse1SynType.NMDA, 
                 self_connect = True)


for s in range(NS):                                 # WTA: S_i -> I
    connect_pops(S_pops[s], I_pop, 
                 p_S_I_base, p_S_I_frac, 
                 dyn1.Dynapse1SynType.AMPA, 
                 dyn1.Dynapse1SynType.NMDA)


for s in range(NS):                                 # WTA: I -> S_i
    connect_pops(I_pop, S_pops[s],
                p_I_S_base, p_I_S_frac,
                dyn1.Dynapse1SynType.GABA_A,
                dyn1.Dynapse1SynType.GABA_B)


if p_I_I_base > 0.0:                                # OPTIONAL INHIBITORY RECURRENCE: I -> I
    connect_pops(I_pop, I_pop,
                    p_I_I_base, p_I_I_frac,
                    dyn1.Dynapse1SynType.GABA_A,
                    dyn1.Dynapse1SynType.GABA_B)


for s in range(NS):                                 # STATE POPULATIONS -> SUMMATION POPULATION: S -> Sum
    connect_pops(S_pops[s], Sum_pop, 
                 p_S_Sum_base, p_S_Sum_frac, 
                 dyn1.Dynapse1SynType.AMPA, 
                 dyn1.Dynapse1SynType.NMDA)


connect_pops(Sum_pop, Sum_pop,                      # SUMMATION POPULATION RECURRENCE
            p_Sum_Sum_base, p_Sum_Sum_frac, 
            dyn1.Dynapse1SynType.AMPA, 
            dyn1.Dynapse1SynType.NMDA,
            self_connect = False)


for s in range(NS):                                 # TRANSITION POPULATION RECURRENCE: T_i -> T_i
    connect_pops(T_pops[s], T_pops[s], 
                 p_T_T_base, p_T_T_frac, 
                 dyn1.Dynapse1SynType.AMPA, 
                 dyn1.Dynapse1SynType.NMDA,
                 self_connect = True)


for t in range(NS):
    # ------------------------------------------------------------
    # TRANSITION POPULATIONS -> PREVIOUS STATE POPULATIONS
    #
    # T_0 -> S_{NS-1}
    # T_1 -> S_0
    # T_2 -> S_1
    # etc.
    # ------------------------------------------------------------

    target_s = (t - 1) % NS

    connect_pops(T_pops[s], S_pops[target_s], 
                p_T_T_base, p_T_T_frac, 
                dyn1.Dynapse1SynType.AMPA, 
                dyn1.Dynapse1SynType.NMDA,
                self_connect = False)


for t in range(NS):                                 # SUM POPULATION -> IT POPULATIONS
    connect_pops(Sum_pop, IT_pops[t],
                p_Sum_IT_base, p_Sum_IT_frac,
                dyn1.Dynapse1SynType.AMPA,
                dyn1.Dynapse1SynType.NMDA)


for t in range(NS):                                 # IT_i -> T_i
    connect_pops(IT_pops[t], T_pops[t],
                p_IT_T_base, p_IT_T_frac,
                dyn1.Dynapse1SynType.GABA_A,
                dyn1.Dynapse1SynType.GABA_B)


for t in range(NS):                                 # IIT_i -> IT_i
    connect_pops(IIT_pops[t], T_pops[t],
                p_IIT_IT_base, p_IIT_IT_frac,
                dyn1.Dynapse1SynType.GABA_A,
                dyn1.Dynapse1SynType.GABA_B)


for s in range(NS):                                 # S_i -> IIT_i
    connect_pops(S_pops[s], IIT_pops[s],
                p_T_S_base, p_T_S_frac ,
                dyn1.Dynapse1SynType.AMPA,
                dyn1.Dynapse1SynType.NMDA)





from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

plt.clf()
plt.close('all')

fig, ax = plt.subplots(figsize=(8, 8))

# ------------------------------------------------------------
# DISCRETE SYNAPSE COLORS
# ------------------------------------------------------------
values = [-2, -1, 0, 1, 2]

colors = [
    "#525763",  # GABA_B  dark grey-blue
    "#acacac",  # GABA_A  light grey
    "#ffffff",  # no connection
    "#93c5fd",  # AMPA    soft blue
    "#2563eb",  # NMDA    deeper blue
]

cmap = ListedColormap(colors)

# Boundaries halfway between integer values
bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap.N)

im = ax.imshow(
    C,
    cmap=cmap,
    norm=norm,
    interpolation='none',
    aspect='equal'
)

# ------------------------------------------------------------
# GRID
# ------------------------------------------------------------
ax.set_xticks(np.arange(N_tot + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(N_tot + 1) - 0.5, minor=True)

ax.grid(
    which='minor',
    color='#e5e7eb',
    linestyle='-',
    linewidth=0.4
)

ax.tick_params(which='minor', bottom=False, left=False)

# ------------------------------------------------------------
# POPULATION BOUNDARIES
# ------------------------------------------------------------
boundary_color = '#111827'

ax.axhline(NE * NS - 0.5, color=boundary_color, linewidth=1.0)
ax.axvline(NE * NS - 0.5, color=boundary_color, linewidth=1.0)

ax.axhline(N_hw - 0.5, color=boundary_color, linewidth=1.0)
ax.axvline(N_hw - 0.5, color=boundary_color, linewidth=1.0)

# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------
ax.set_xlabel("target neuron")
ax.set_ylabel("source neuron")
ax.set_title("Connectivity matrix")

# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------
legend_elements = [
    Patch(facecolor="#93c5fd", edgecolor='none', label='AMPA'),
    Patch(facecolor="#2563eb", edgecolor='none', label='NMDA'),
    Patch(facecolor="#a3a3a3", edgecolor='none', label='GABA_A'),
    Patch(facecolor="#6b7280", edgecolor='none', label='GABA_B'),
]

ax.legend(
    handles=legend_elements,
    loc='upper right',
    frameon=True,
    framealpha=1.0,
    edgecolor='#e5e7eb'
)

plt.tight_layout()
plt.show()



# ------------------------------------------------------------
# APPLY CONFIGURATION
# ------------------------------------------------------------
new_config = net_gen.make_dynapse1_configuration()
model.apply_configuration(new_config)

set_params(model, param_group=param_group)

# ------------------------------------------------------------
# PREPARE POISSON INPUT
# ------------------------------------------------------------
global_poisson_gen_ids = ut.get_global_id_list(spikegen_ids)
poisson_gen            = model.get_poisson_gen()
poisson_gen.set_chip_id(0)

# set initial rates to zero
for gid in global_poisson_gen_ids:
    poisson_gen.write_poisson_rate_hz(gid, 0)

# ------------------------------------------------------------
# MONITOR NEURONS
# ------------------------------------------------------------

monitored_neurons = (
    S_ids
    + T_ids
    + IT_ids
    + IIT_ids
    + I_ids
    + Sum_ids
)

graph, filter_node, sink_node = ut.create_neuron_select_graph(
    model,
    monitored_neurons
)

graph.start()

# clear buffer
sink_node.get_events()
api.reset_timestamp()
