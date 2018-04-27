###############################import###############################
import operator
import math
import random
import pandas as pd
import numpy
import time
import os
import dill
from sympy import *
from sympy.parsing import sympy_parser

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from scoop import IS_ORIGIN, futures

from pandas import DataFrame

from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing as mp

###################################################################

###################### Define functions############################
def safeDiv(left, right):
    if right==0:
        return 1
    else:
        return left/right

    
def sqrt(left):
    if left>=0:
        return left**0.5
    else:
        return 0
    
def expe(left):
    if left>=0:
        return numpy.exp(-left)
    else:
        return 0




pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(safeDiv,2)
try:
    if not scoop.IS_RUNNING:
        pset.addEphemeralConstant("rand", random.randint(1,5))
except:
    print("ops")
# try:
#     pset.addEphemeralConstant("rand", lambda: random.randint(1,5))
# except:
#     print("OPPS")

pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')
# pset.renameArguments(ARG2='c1')
# pset.renameArguments(ARG3='c2')
# pset.renameArguments(ARG4='c3')
# pset.renameArguments(ARG5='c4')




###################################################################
pool = mp.Pool()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("map", futures.map)

#########################input DATA#################################

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Evaluate the mean squared error between the expression
    sqerrors = ((func(x,y) - r)**2 for x,y,r in points)

    try:
        return max(sqerrors),
    except IndexError:
        return max(sqerrors),
    
# toolbox.register("map", pool.map)


g = pd.read_csv('2d.csv')
p = g['x']
q = g['y']
r = g['f']

pts=[]

for i in range(len(r)):
    pts.append((p[i],q[i],r[i]))
    
    
toolbox.register("evaluate", evalSymbReg, points=pts)
toolbox.register("select", tools.selTournament, tournsize=100)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))


def main():
    pop = toolbox.population(n=10000)
    hof = tools.HallOfFame(1)
   
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    
    #############################################C.O, Muta,Gener##########
    try:
        pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.8, 500, stats=mstats,
                                       halloffame=hof, verbose=True)
        print (hof[0])
#         plot_pred(hof)
        return hof
    
    except MemoryError:
        print (hof[0])
#         plot_pred(hof)
        return hof
    
    except KeyboardInterrupt:
        print (hof[0])
#         plot_pred(hof)
        return hof



if __name__ == "__main__":          
    hof = main()