#!/usr/bin/env python
'''
This script demonstrates the use of a genetic algorithm (using Pyevolve) to
generate a resource curve, which controls the transfer of water between
two reservoirs. The pumping cost is minimized, while still finding a solution
which does not result in any failures to supply demand.
'''

import os
import pandas as pd

from pywr.core import *

RESERVOIR_MAX_VOLUME = 20000
RESERVOIR_INITIAL_VOLUME = RESERVOIR_MAX_VOLUME * 0.8
EMERGENCY_STORAGE_VOLUME = RESERVOIR_MAX_VOLUME * 0.15
TRANSFER_COST = 2.0
LINK_MAX_FLOW = 10.0

def interpolate_profile(xp, yp):
    y = np.interp(np.arange(0, 367, dtype=int), xp, yp)
    return y

def create_model():
    model = Model()
    
    df = pandas.read_csv(os.path.join(os.path.dirname(__file__), 'AQTF5CC.csv'), index_col=0, parse_dates=True, dayfirst=True)
    df = df['Flow']
    ts1 = Timeseries(df)
    model.data['ts1'] = ts1
    
    catchment1 = Catchment(model=model, name='catch1', position=(-2,3))
    catchment2 = Catchment(model=model, name='catch2', position=(2,3))

    catchment1.properties['flow'] = ts1
    catchment2.properties['flow'] = ts1

    # abstractions are required as model doesn't currently support river
    # flowing directly into the reservoir
    abs1 = RiverAbstraction(model=model, max_flow=9999, position=(-2,1))
    abs2 = RiverAbstraction(model=model, max_flow=9999, position=(2,1))

    reservoir1 = Reservoir(model=model, name='res1', position=(-1,1))
    reservoir2 = Reservoir(model=model, name='res2', position=(1,1))

    for reservoir in (reservoir1, reservoir2):
        # model doesn't do emergency storage yet, so model it as a reduction in max volume
        reservoir.properties['max_volume'] = Parameter(RESERVOIR_MAX_VOLUME - EMERGENCY_STORAGE_VOLUME)
        reservoir.properties['current_volume'] = Parameter(RESERVOIR_INITIAL_VOLUME)

    # model doesn't do compensation flow yet, so add it as a demand instead
    # for our purposes this has the same effect
    demand1 = Demand(model=model, name='demand1', demand=81.0 + 5, position=(-1,2))
    demand2 = Demand(model=model, name='demand2', demand=100.0 + 5, position=(1,2))

    term1 = Terminator(model=model, name='term1', position=(0,0))

    catchment1.connect(abs1)
    catchment2.connect(abs2)
    abs1.connect(reservoir1)
    abs2.connect(reservoir2)
    abs1.connect(term1)
    abs2.connect(term1)
    reservoir1.connect(demand1)
    reservoir2.connect(demand2)
    
    # the link between the two reservoirs
    link1 = Link(model, name='link1', position=(0,1))
    reservoir1.connect(link1)
    link1.connect(reservoir2)

    model.check()

    return model

def reset_model(model):
    model.timestamp = pd.to_datetime('1990-01-01')
    nodes = dict([(n.name, n) for n in model.nodes()])
    nodes['res1'].properties['current_volume'] = Parameter(RESERVOIR_INITIAL_VOLUME)
    nodes['res2'].properties['current_volume'] = Parameter(RESERVOIR_INITIAL_VOLUME)

def run(model, xp, yp):
    reset_model(model)

    nodes = dict([(n.name, n) for n in model.nodes()])
    
    res1 = nodes['res1']
    res2 = nodes['res2']
    
    resource_curve = interpolate_profile(xp, yp)
    
    def flow_func(self, timestamp):
        '''Calculate transfer between two reservoirs'''
        current_volume = res2.properties['current_volume']._value
        percentage = resource_curve[timestamp.dayofyear]
        if current_volume < RESERVOIR_MAX_VOLUME * percentage:
            return LINK_MAX_FLOW
        else:
            return 0.0
    nodes['link1'].properties['max_flow'] = ParameterFunction(nodes['link1'], flow_func)
    
    # run the model for each day in the timeseries
    cost = 0.0
    days = len(model.data['ts1'].df)
    for n in range(days):
        res = model.step()
        
        # calculate total water supplied to the demands
        total_supplied = 0.0
        try:
            total_supplied += res[3][(nodes['res1'], nodes['demand1'])]
        except KeyError:
            pass
        try:
            total_supplied += res[3][(nodes['res2'], nodes['demand2'])]
        except KeyError:
            pass

        if total_supplied != 191:
            #print(total_supplied)
            cost += (191-total_supplied) * 100.0
        
        # update the cumulative pumping cost
        try:
            cost += res[3][(nodes['res1'], nodes['link1'])] * TRANSFER_COST
        except KeyError:
            pass
    
    return (n, cost)

model = create_model()

# get day in year for first day in each month
month_days = []
for n in range(1, 12+1, 3):
    month_days.append(pd.to_datetime('2015-{}-1'.format(n)).dayofyear)

def eval_func(chromosome):
    #chromosome = [1.0]*len(month_days)
    # reset the model to it's inital state
    reset_model(model)
    # interpolate the resource curve
    xp = month_days + [367,]
    yp = chromosome[:] + [chromosome[0],]
    #yp = [1.0]*13
    # evaluate the model
    days, total_cost = run(model, xp, yp)
    #print(total_cost, ['{:.2f}'.format(c) for c in chromosome[:]])
    return total_cost

from pyevolve import G1DList, GSimpleGA, Consts, DBAdapters, Initializators, Mutators

# configure the GA
genome = G1DList.G1DList(len(month_days))
genome.setParams(rangemin=0.0, rangemax=1.0, gauss_sigma=0.05)
genome.initializator.set(Initializators.G1DListInitializatorReal)
genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
genome.evaluator.set(eval_func)
ga = GSimpleGA.GSimpleGA(genome)
ga.setGenerations(100)
ga.setPopulationSize(10)
ga.setCrossoverRate(0.8)
ga.setMutationRate(0.1)
#ga.setElitism(True)
#ga.setElitismReplacement(1)
ga.setMultiProcessing(flag=True, full_copy=True)
ga.setMinimax(Consts.minimaxType["minimize"])

# log to sqlite database
# to view results, try:
# $ python venv/bin/pyevolve_graph.py -i ex1 -f pyevolve.db -3
sqlite_adapter = DBAdapters.DBSQLite(identify='ga1')
ga.setDBAdapter(sqlite_adapter)

print('Running genetic algorithm...')
ga.evolve(freq_stats=1)

print ga.bestIndividual()

# display the resource curve for the best individual
import matplotlib.pyplot as plt
chromosome = ga.bestIndividual().genomeList
plt.plot(range(1, 14), chromosome + [chromosome[0]])
plt.xlim(1, 13)
plt.show()
