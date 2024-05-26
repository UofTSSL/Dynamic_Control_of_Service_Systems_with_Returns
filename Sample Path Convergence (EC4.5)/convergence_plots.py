import SimFunctions
import SimRNG 
import SimClasses
import numpy as np
import scipy.stats as stats
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from itertools import product
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os

# Change to the directory you want the ouput to be saved in
directory = "csvs\\"
log_file = "sims.log"
knn_file = "knn_linear_2023-08-23_20-03-20.pkl"


T_MAX = 365
MAX_STEP = 0.02


with open(knn_file, "rb") as f:
    knn = pickle.load(f)
    
class Experiment:

    def g1(self, s):
        return self.hold_cost * s + (self.return_cost * self.p_eq + self.interv_cost(self.p_eq)) / (1 - self.p_eq)

    def g2(self, s):
        return self.hold_cost / self.return_rate * \
               (np.exp(-self.return_rate * s) + self.return_rate * s - 1) + (self.return_cost + self.interv_cost(self.p_eq)) / (1 - self.p_eq)

    def p(self, tau):
        return optimize.minimize_scalar(
            lambda q: self.interv_cost(q) + self.g2(tau) * q,
            bounds=(self.p_l, self.p_h), method="bounded"
        ).x

    def pol(self, costate):
        return optimize.minimize_scalar(
            lambda q: self.interv_cost(q) + costate * q,
            bounds=(self.p_l, self.p_h), method="bounded"
        ).x

    def y_coef(self, tau):
        return self.hold_cost * (1 - np.exp(-self.return_rate * tau))

    def const(self, tau):
        prob = self.p(tau)
        return -self.J_eq + (self.arrival_rate - self.service_rate * self.server_num) * self.g1(tau) \
               + self.service_rate * self.server_num * (self.interv_cost(prob) + self.g2(tau) * prob)

    def optimal_policy(self, x, y):
        return knn.predict([[x,y]])[0]

    def eq_policy(self, x, y):
        return self.p_eq

    def simple_policy(self, x, y):
        if x < self.server_num:
            return self.p_eq
        return self.p_l


    def solve_trajectory(self, y0):
        def empty_ward(t, vec):
            return vec[0]

        empty_ward.terminal = True
        empty_ward.direction = -1.

        def empty_orbit(t, vec):
            return vec[1]

        empty_orbit.terminal = True
        empty_orbit.terminal = -1.

        def grad(t, vec):
            x, y, c1, c2 = vec
            p = self.pol(c2)
            dx = self.arrival_rate + self.return_rate * y - self.service_rate * min(x, self.server_num)
            dy = -self.return_rate * y + self.service_rate * min(x, self.server_num) * p
            dc1 = self.service_rate * (c1 - self.interv_cost(p) - c2 * p) if x < self.server_num else -self.hold_cost
            dc2 = self.return_rate * (-self.return_cost - c1 + c2)

            return -np.array([dx, dy, dc1, dc2])
        return integrate.solve_ivp(grad, [0, T_MAX], [self.server_num, y0, self.g1_eq, self.g2_eq], max_step=MAX_STEP, events=[empty_ward, empty_orbit])

    def __init__(self, **kwargs):

        self.params = kwargs

        self.reps = kwargs.get("reps", 100)
        self.run_length = kwargs.get("run_length", 365)
        self.warm_up = kwargs.get("warm_up", 0)#1000)
        self.metrics = [[], [], []]

        self.server_num = kwargs.get('N', 50)
        self.arrival_rate = kwargs.get('lambda', 9.5)
        self.mean_tba = 1 / self.arrival_rate
        self.return_rate = kwargs.get('nu', 1/15)
        self.mean_ttr = 1 / self.return_rate
        self.service_rate = kwargs.get('mu', 1/4)
        self.mean_st = 1 / self.service_rate
        self.p_h = kwargs.get('p_h', 0.2)
        self.p_l = kwargs.get('p_l', 0.1)

        self.fluctuation = kwargs.get('fluctuation', 0)
        self.period = kwargs.get('period', 1)

        self.hold_cost = kwargs.get('h', 0.2)
        self.return_cost = kwargs.get('r', 1.0)

        self.interv_cost_str = kwargs.get('C', "lambda p: 10 * 0.5 * (0.2-p)")
        self.interv_cost = eval(self.interv_cost_str)

        self.x_0 = kwargs.get('x_0', 75)
        self.y_0 = kwargs.get('y_0', 75)
        self.scale = kwargs.get('scale', 50)

        self.p_eq = optimize.minimize_scalar(
            lambda p: (self.return_cost * p + self.interv_cost(p)) / (1 - p),
            bounds=(self.p_l, self.p_h), method="bounded"
        ).x

    def run(self, stream=0):
        sim = Simulation(self, policy=self.optimal_policy)
        self.times, self.Xs, self.Ys = sim.simulate(stream)

    def save(self):
        timestamp = datetime.now()
        time_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        path = directory + "simulation_" + time_str + ".csv"
        frame = pd.DataFrame({
                't': self.times,
                'x': self.Xs,
                'y': self.Ys})
        frame.to_csv(path)

        file = open(directory + log_file, "a")
        file.write("\n" + time_str + "," + str(self.params))
        file.close()


# Given reference point i, use the following streams for RNGs
# i+1: External arrivals ~Exp(Lambda)
# i+2: Service times ~Exp(mu)
# i+3: Return times ~Exp(nu)
# i+4: Returns ~Bernoulli(p)


class Simulation:

    def __init__(self, experiment, warm_up=0, **kwargs):
        self.experiment = experiment
        self.policy = kwargs.get("policy", lambda x, y : experiment.p_h)
        self.stream = 1

        self.queue = SimClasses.FIFOQueue()
        self.wait_stat = SimClasses.DTStat()
        self.interv_stat = SimClasses.DTStat()
        self.interv_cost_stat = SimClasses.DTStat()
        self.returns_stat = SimClasses.DTStat()
        self.congestion_stat = SimClasses.CTStat()
        self.orbit_stat = SimClasses.CTStat()
        self.server = SimClasses.Resource()
        self.server.SetUnits(self.experiment.server_num * self.experiment.scale // 50)
        self.calendar = SimClasses.EventCalendar()

        self.is_congested_stat = SimClasses.CTStat()
        self.is_intervening_stat = SimClasses.CTStat()

        self.X = self.experiment.x_0 * self.experiment.scale // 50
        self.Y = self.experiment.y_0 * self.experiment.scale // 50

        self.ct_stats = []
        self.dt_stats = []
        self.queues = []
        self.resources = []

        self.dt_stats.append(self.interv_stat)
        self.dt_stats.append(self.interv_cost_stat)
        self.dt_stats.append(self.returns_stat)
        self.ct_stats.append(self.congestion_stat)
        self.ct_stats.append(self.orbit_stat)

        self.ct_stats.append(self.is_congested_stat)
        self.ct_stats.append(self.is_intervening_stat)

        self.queues.append(self.queue)
        self.resources.append(self.server)
        
        self.times = []
        self.Xs = []
        self.Ys = []

    def record(self):
        self.times.append(SimClasses.Clock)
        self.Xs.append(self.X)
        self.Ys.append(self.Y)

    def NSPP(self):
        PossibleArrival = SimClasses.Clock + SimRNG.Expon(
            self.experiment.mean_tba / (1 + self.experiment.fluctuation) / self.experiment.scale * 50, self.stream)
        while SimRNG.Uniform(0, 1, self.stream) >= (1.0 + self.experiment.fluctuation * np.sin(2*np.pi * PossibleArrival / self.experiment.period / self.experiment.scale*50)) / (1 + self.experiment.fluctuation):
            PossibleArrival = PossibleArrival + SimRNG.Expon(self.experiment.mean_tba / (1 + self.experiment.fluctuation) / self.experiment.scale * 50, self.stream)
        nspp = PossibleArrival - SimClasses.Clock
        return nspp

    def Arrival(self):
        SimFunctions.Schedule(self.calendar, "Arrival", self.NSPP())
        # SimFunctions.Schedule(self.calendar,"Arrival",SimRNG.Expon(self.experiment.mean_tba/self.experiment.scale*50, self.stream))
        # SimFunctions.Schedule(self.calendar,"Arrival",SimRNG.Expon(self.experiment.mean_tba/self.experiment.scale*50, self.stream+1))
        customer = SimClasses.Entity()
        self.queue.Add(customer)

        self.X += 1
        self.congestion_stat.Record(max(self.X-self.experiment.server_num*self.experiment.scale/50, 0))
        self.is_congested_stat.Record(1 if self.X > self.experiment.server_num*self.experiment.scale/50 else 0)
        self.is_intervening_stat.Record(self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale))

        if self.server.Busy < self.experiment.server_num * self.experiment.scale/50:
            self.server.Seize(1)
            SimFunctions.Schedule(self.calendar, "EndOfService", SimRNG.Expon(self.experiment.mean_st, self.stream))
            # SimFunctions.Schedule(self.calendar,"EndOfService",SimRNG.Expon(self.experiment.mean_st,self.stream+2))
            
        self.record()

    def EndOfService(self):
        DepartingCustomer = self.queue.Remove()
        self.wait_stat.Record(SimClasses.Clock - DepartingCustomer.CreateTime)
        # if there are customers waiting
        if self.queue.NumQueue() >= self.experiment.server_num * self.experiment.scale/50:
            SimFunctions.Schedule(self.calendar, "EndOfService", SimRNG.Expon(self.experiment.mean_st, self.stream))
            # SimFunctions.Schedule(self.calendar,"EndOfService",SimRNG.Expon(self.experiment.mean_st,self.stream+2))
        else:
            self.server.Free(1)

        p = self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale)
        self.interv_stat.Record(p)
        self.interv_cost_stat.Record(self.experiment.interv_cost(p))
        # if SimRNG.Uniform(0,1,self.stream+4) < p:
        if SimRNG.Uniform(0, 1, self.stream) < p:
            self.Y += 1
            self.orbit_stat.Record(self.Y)
            SimFunctions.Schedule(self.calendar, "Return", SimRNG.Expon(self.experiment.mean_ttr, self.stream))
            # SimFunctions.Schedule(self.calendar,"Return",SimRNG.Expon(self.experiment.mean_ttr, self.stream+3))

        self.X -= 1
        self.congestion_stat.Record(max(self.X - self.experiment.server_num * self.experiment.scale/50, 0))
        self.is_congested_stat.Record(1 if self.X > self.experiment.server_num*self.experiment.scale/50 else 0)
        self.is_intervening_stat.Record(self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale))
        
        self.record()

    def Return(self):
        self.orbit_stat.Record(self.Y)
        customer = SimClasses.Entity()
        self.queue.Add(customer)

        self.X += 1
        self.Y -= 1
        self.returns_stat.Record(1)
        self.congestion_stat.Record(max(self.X-self.experiment.server_num*self.experiment.scale/50, 0))
        self.is_congested_stat.Record(1 if self.X > self.experiment.server_num*self.experiment.scale/50 else 0)
        self.is_intervening_stat.Record(self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale))
        self.orbit_stat.Record(self.Y)

        if self.server.Busy < self.experiment.server_num*self.experiment.scale/50:
            self.server.Seize(1)
            SimFunctions.Schedule(self.calendar,"EndOfService",SimRNG.Expon(self.experiment.mean_st,2))
            
        self.record()

    def initialize(self):

        SimFunctions.SimFunctionsInit(self.calendar,self.queues,self.ct_stats,self.dt_stats,self.resources)

        for i in range(self.X):
            self.queue.Add(SimClasses.Entity())
            if self.server.Busy < self.experiment.server_num*self.experiment.scale/50:
                self.server.Seize(1)
                SimFunctions.Schedule(self.calendar, "EndOfService",
                                      SimRNG.Expon(self.experiment.mean_st, self.stream))
                # SimFunctions.Schedule(self.calendar, "EndOfService",
                #                       SimRNG.Expon(self.experiment.mean_st, self.stream + 2))

        for j in range(self.Y):
            SimFunctions.Schedule(self.calendar, "Return", SimRNG.Expon(self.experiment.mean_ttr, self.stream))
            # SimFunctions.Schedule(self.calendar, "Return", SimRNG.Expon(self.experiment.mean_ttr, self.stream + 3))

        self.congestion_stat.Record(max(self.X - self.experiment.server_num*self.experiment.scale/50, 0))
        self.is_congested_stat.Record(1 if self.X > self.experiment.server_num*self.experiment.scale/50 else 0)
        self.is_intervening_stat.Record(self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale))
        self.orbit_stat.Record(self.Y)

        SimFunctions.Schedule(self.calendar, "Arrival", self.NSPP())
        # SimFunctions.Schedule(self.calendar, "Arrival", SimRNG.Expon(self.experiment.mean_tba / self.experiment.scale * 50, self.stream))
        # SimFunctions.Schedule(self.calendar, "Arrival", SimRNG.Expon(self.experiment.mean_tba/self.experiment.scale*50, self.stream + 1))
        SimFunctions.Schedule(self.calendar, "EndSimulation", self.experiment.run_length + self.experiment.warm_up)
        SimFunctions.Schedule(self.calendar, "ClearIt", self.experiment.warm_up)
        
        self.record()

    def simulate(self, stream=0):
        self.stream = stream
        self.initialize()

        next_event = self.calendar.Remove()
        SimClasses.Clock = next_event.EventTime
        while next_event.EventType != "EndSimulation":
            if next_event.EventType == "Arrival":
                self.Arrival()
            elif next_event.EventType == "EndOfService":
                self.EndOfService()
            elif next_event.EventType == "Return":
                self.Return()
            elif next_event.EventType == "ClearIt":
                SimFunctions.ClearStats(self.ct_stats, self.dt_stats)

            next_event = self.calendar.Remove()
            SimClasses.Clock = next_event.EventTime

        total_holding_cost = self.experiment.hold_cost * self.congestion_stat.Area
        total_readmission_cost = self.experiment.return_cost * self.returns_stat.Sum
        total_intervention_cost = self.interv_cost_stat.Sum
        avg_intervention = self.interv_stat.Mean()
        combined_cost = total_holding_cost + total_readmission_cost + total_intervention_cost

        avg_congested_time = self.is_congested_stat.Mean()
        avg_intervention_time = self.is_intervening_stat.Mean()


        return self.times, self.Xs, self.Ys

# Run simulations and save the ouputs as csvs
def main():

    inits = [[80,60],[20,100], [20,80], [5,65]]

    for init in inits:
        for i in range(30):
            exp = Experiment(**{'scale': 500,
                                'lambda': 9.5,
                                'nu': 1 / 15,
                                'mu': 1 / 4,
                                'h': 0.25,
                                'C': "lambda p: 10*{}*(0.2-p)".format(0.5),
                                'x_0': init[0],
                                'y_0': init[1],
                                'reps': 1,
                                'fluctuation': 0,
                                'period': 1,
                                'warm_up': 0,
                                'run_length': 365,
                                })
            exp.run(i+10)
            exp.save()

main()

def plot_average_and_trajectories(directory, knn_model, time_bins):
    # Get average sample path
    avg_df = average_sample_path_from_csvs(directory, time_bins)

    fig, ax1 = plt.subplots()

    plot_length = min(len(avg_df['time_bin']), len(avg_df['avg_x']), len(avg_df['avg_y']))

    # Plot average trajectories
    ax1.plot(avg_df['time_bin'][:plot_length], avg_df['avg_x'][:plot_length]/500, label='Sample Path Avg (Needy)', color='grey')
    ax1.plot(avg_df['time_bin'][:plot_length], avg_df['avg_y'][:plot_length]/500, label='Sampel Path Avg (Content)', color='black')

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Plot KNN trajectories for a subset (to avoid clutter)
    count = 0
    for csv_file in csv_files[:30]:  # only plotting the first 5 as an example
        path = os.path.join(directory, csv_file)
        df = pd.read_csv(path)
        
        if not (np.isnan(df.x.iloc[0]) or np.isnan(df.y.iloc[0])):
            if count ==1: break
            else: 
                sol = solve_trajectory(df.x.iloc[0]/10, df.y.iloc[0]/10, lambda a, b: knn_model.predict([[a, b]])[0])
                ax1.plot(sol.t, sol.y[0]/50, color='grey', alpha=0.5,linestyle='--', label='Fluid Traj. (Needy)')  # Needy State
                ax1.plot(sol.t, sol.y[1]/50, color='black', alpha=0.5,linestyle='--', label='Fluid Traj. (Content)')  # Content State
                count =1

    ax1.set_xlabel('Time')
    ax1.set_ylabel('State Value')
    ax1.legend()
    time_bins = np.linspace(0, 100, 100)  # from 0 to 100 with 50 bins
    plt.xlim(0, 100)  # Adjusts the x-axis to only go from 0 to 100
    plt.savefig('cvg_2.png', dpi=600)
    plt.show()

# Define the directory path where your CSV files are located
directory_path = "./csvs"  # replace with your actual path

# The optimal policy to plot the trajectories
knn_model = knn  

# Define the time bins for discretization. Adjust as needed.
time_bins = np.linspace(0, 100, 500)  # for example: from 0 to 100 with 50 bins

# Call the function to plot
plot_average_and_trajectories(directory_path, knn_model, time_bins)