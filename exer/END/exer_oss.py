def plot_pred(hof):
    func = toolbox.compile(expr=hof[0])
    y_hat = []
    y = []
    y_r = []
    for a, i in enumerate(t):
        y_hat.append(func(i))
        y.append(p[a])
        y_r.append(p_r[a])
        
    from matplotlib import pyplot as plt
    plt.plot(t,y)
    
    plt.plot(t, y_hat)
    plt.plot(t, y_r)
    plt.xlabel('time')  # x-axis
    plt.ylabel('x(t)')  # y-axis
    plt.title('Data')  # title
    plt.grid()  # grid
    plt.legend(['with error','predicted','real'])
    plt.show()  # plot show
    

###############################import###############################
import operator
import math
import random
import pandas as pd
import numpy
import time


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from scoop import futures
import multiprocessing as mp
import matplotlib.pyplot as plt



###################### Define functions############################
def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
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

    
    
random.seed(time.time())
ran1 = int(random.random()*10)
ran2 =int(random.random()*10)

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
#pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(expe, 1)
#set.addPrimitive(safeDiv, 2)
pset.addPrimitive(sqrt,1)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
try:
    pset.addEphemeralConstant("rand", random.randint(1,5))
except:
    print("")
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

pool = mp.Pool()


'''g = pd.read_csv('data.csv')
p = g['x']
t = list(numpy.arange(0,1400))
f = {}
for i in range(len(t)):
    f[t[i]] = p[i]'''



def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Evaluate the mean squared error between the expression
    sqerrors = ((func(x) - f[x])**2 for x in points)

    try:
        return max(sqerrors),
    except IndexError:
        return max(sqerrors),
    
toolbox.register("map", pool.map)
g = pd.read_csv('os.csv')
p = g['x']
g_r = pd.read_csv('os_r.csv')
p_r = g_r['x']

t = list(numpy.arange(0,15,0.01))
f = {}
for i in range(len(t)):
    f[t[i]] = p[i]
    
ff = {}
for i in range(len(t)):
    ff[t[i]] = p_r[i]
toolbox.register("evaluate", evalSymbReg, points=t)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))





#########################input DATA#################################



###################################################################





def main():
    pop = toolbox.population(n=7000)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    
    #############################################C.O, Muta,Gener##########
    try:
        pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.8, 1000, stats=mstats,
                                       halloffame=hof, verbose=True)
        print (hof[0])
        
        plot_pred(hof)

    except MemoryError:
        print (hof[0])
        plot_pred(hof)

    except KeyboardInterrupt:
        print (hof[0])
        plot_pred(hof)


    


if __name__ == "__main__":
    main()
