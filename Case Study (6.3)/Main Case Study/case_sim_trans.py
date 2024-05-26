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

directory = "outputs_case_primary\\"
log_file = "sims.log"


T_MAX = 365
MAX_STEP = 0.02


p_h = 0.141
p_l = 0.081
M = 1110

def C(p):
    return M * (p_h - p) / (p_h - p_l)




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
        if x <= self.server_num:
            if y <= (self.service_rate * self.server_num - self.arrival_rate) / self.return_rate:
                return self.p_eq
            else:
                return self.knn.predict(np.array([[x,y]]))[0]
        dists = self.hold_cost * (x - self.server_num) + self.y_coefs * y + self.constants
        if np.all(dists > 0):
            return self.p_l
        return self.policies[np.argmax(dists < 0)]

    def eq_policy(self, x, y):
        return self.p_eq

    def simple_policy(self, x, y):
        if x < self.server_num:
            return self.p_eq
        return self.p_l

    def safety_policy(self, x, y):
        return self.optimal_policy(x+np.log(self.scale)*50/self.scale, y+np.log(self.scale)*50/self.scale)


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
        self.metrics = [[], [], [], []]

        self.server_num = kwargs.get('N', 50)
        self.arrival_rate = kwargs.get('lambda', 9.5)
        self.mean_tba = 1 / self.arrival_rate
        
        self.return_rate = kwargs.get('nu', 1/15)
        self.mean_ttr = 1 / self.return_rate
        self.trunc_return_rate = 1 / 25
        
        self.service_rate = kwargs.get('mu', 1/4)
        self.mean_st = 1 / self.service_rate
        self.lognormal_mu = 1.38
        self.lognormal_sigma = 0.83

        self.stream = kwargs.get("stream", np.random.randint(100))

        
        self.p_h = p_h
        self.p_l = p_l

        self.fluctuation = kwargs.get('fluctuation', 0)
        self.period = kwargs.get('period', 1)

        self.hold_cost = kwargs.get('h', 0.2)
        self.return_cost = kwargs.get('r', 1.0)

        self.M = kwargs.get('M', 150)
        self.interv_cost = C
        
        


        self.x_0 = kwargs.get('x_0', 75)
        self.y_0 = kwargs.get('y_0', 75)
        self.scale = kwargs.get('scale', 50)

        self.p_eq = optimize.minimize_scalar(
            lambda p: (self.return_cost * p + self.interv_cost(p)) / (1 - p),
            bounds=(self.p_l, self.p_h), method="bounded"
        ).x
        self.J_eq = self.arrival_rate * (self.return_cost * self.p_eq + self.interv_cost(self.p_eq)) / (1 - self.p_eq)
        self.g1_eq = (self.return_cost * self.p_eq + self.interv_cost(self.p_eq)) / (1 - self.p_eq)
        self.g2_eq = (self.return_cost + self.interv_cost(self.p_eq)) / (1 - self.p_eq)

        
        with open('pickle_{}.pkl'.format(self.hold_cost), 'rb') as f:
            self.policies, self.constants, self.y_coefs, self.knn = pickle.load(f)

        self.policy_variants = [self.eq_policy, self.simple_policy, self.optimal_policy]


    def run(self):
        for j, pol in enumerate(self.policy_variants):
            # SimRNG.InitializeRNSeed()
            for i in range(self.reps):
                sim = Simulation(self, policy=pol, stream=self.stream)
                results = sim.simulate(stream=self.stream)
                self.metrics[j].append(list(results))

    def save(self):
        timestamp = datetime.now()
        time_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        path = directory + "simulation_" + time_str + ".csv"
        headers = ['h_cost', 'r_cost', 'C_cost', 'total_cost', 'mean_p', 'mean_wait', 'mean_queue', 'mean_orbit', 'mean_server', 'x_T', 'y_T', 'mean_p_time', 'mean_congested_time']
        eq_frame = pd.DataFrame(self.metrics[0], columns=headers)
        simple_frame = pd.DataFrame(self.metrics[1], columns=headers)
        opt_frame = pd.DataFrame(self.metrics[2], columns=headers)
        safety_frame = pd.DataFrame(self.metrics[3], columns=headers)
        frame = pd.concat([eq_frame, simple_frame, opt_frame, safety_frame], axis=1)
        frame.to_csv(path)

        file = open(directory + log_file, "a")
        file.write("\n" + time_str + "," + str(self.params))
        file.close()



class Simulation:

    def __init__(self, experiment, warm_up=0, **kwargs):
        self.experiment = experiment
        self.policy = kwargs.get("policy", lambda x, y : experiment.p_h)
        self.stream = kwargs.get("stream", 0)

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

    def NSPP(self):
        arrival_rate = 1/self.experiment.mean_tba
        weekend_ratio = 0.9
        weekday_rate = arrival_rate * (1+0.4*(1-weekend_ratio)) # arrival_rate / (5/7 + 2/7*weekend_ratio)
        weekend_rate = arrival_rate * weekend_ratio # weekday_rate * weekend_ratio
        max_rate = weekday_rate * (1+self.experiment.fluctuation)

        def varying_rate(t):
            daily_rate = weekday_rate if t % 7 < 5 else weekend_rate
            return daily_rate * (1 + self.experiment.fluctuation * np.sin(-2*np.pi*t))


        PossibleArrival = SimClasses.Clock + SimRNG.Expon(1/max_rate, self.stream)
        while SimRNG.Uniform(0, 1, self.stream) >= varying_rate(PossibleArrival) / max_rate:
            PossibleArrival = PossibleArrival + SimRNG.Expon(1/max_rate, self.stream)
        nspp = PossibleArrival - SimClasses.Clock
        return nspp
    
    

    def Arrival(self):
        SimFunctions.Schedule(self.calendar, "Arrival", self.NSPP())
        customer = SimClasses.Entity()
        self.queue.Add(customer)

        self.X += 1
        self.congestion_stat.Record(max(self.X-self.experiment.server_num*self.experiment.scale/50, 0))
        self.is_congested_stat.Record(1 if self.X > self.experiment.server_num*self.experiment.scale/50 else 0)
        self.is_intervening_stat.Record(self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale))

        if self.server.Busy < self.experiment.server_num * self.experiment.scale/50:
            self.server.Seize(1)
            SimFunctions.Schedule(self.calendar, "EndOfService",
                                  SimRNG.Lognorm(self.experiment.lognormal_mu, self.experiment.lognormal_sigma, self.stream))

    def EndOfService(self):
        DepartingCustomer = self.queue.Remove()
        self.wait_stat.Record(SimClasses.Clock - DepartingCustomer.CreateTime)
        # if there are customers waiting
        if self.queue.NumQueue() >= self.experiment.server_num * self.experiment.scale/50:
            SimFunctions.Schedule(self.calendar, "EndOfService",
                                  SimRNG.Lognorm(self.experiment.lognormal_mu, self.experiment.lognormal_sigma, self.stream))
        else:
            self.server.Free(1)

        p = self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale)
        self.interv_stat.Record(p)
        self.interv_cost_stat.Record(self.experiment.interv_cost(p))
        if SimRNG.Uniform(0, 1, self.stream) < p:
            self.Y += 1
            self.orbit_stat.Record(self.Y)
            SimFunctions.Schedule(self.calendar, "Return", SimRNG.TruncExpon(1/self.experiment.trunc_return_rate, 30, self.stream))

        self.X -= 1
        self.congestion_stat.Record(max(self.X - self.experiment.server_num * self.experiment.scale/50, 0))
        self.is_congested_stat.Record(1 if self.X > self.experiment.server_num*self.experiment.scale/50 else 0)
        self.is_intervening_stat.Record(self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale))

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
            SimFunctions.Schedule(self.calendar,"EndOfService",
                                  SimRNG.Lognorm(self.experiment.lognormal_mu, self.experiment.lognormal_sigma, self.stream))

    def initialize(self):

        SimFunctions.SimFunctionsInit(self.calendar,self.queues,self.ct_stats,self.dt_stats,self.resources)

        for i in range(self.X):
            self.queue.Add(SimClasses.Entity())
            if self.server.Busy < self.experiment.server_num*self.experiment.scale/50:
                self.server.Seize(1)
                SimFunctions.Schedule(self.calendar, "EndOfService",
                                      SimRNG.Lognorm(self.experiment.lognormal_mu, self.experiment.lognormal_sigma, self.stream))

        for j in range(self.Y):
            SimFunctions.Schedule(self.calendar, "Return", SimRNG.TruncExpon(1/self.experiment.trunc_return_rate, 30, self.stream))

        self.congestion_stat.Record(max(self.X - self.experiment.server_num*self.experiment.scale/50, 0))
        self.is_congested_stat.Record(1 if self.X > self.experiment.server_num*self.experiment.scale/50 else 0)
        self.is_intervening_stat.Record(self.policy(self.X * 50 / self.experiment.scale, self.Y * 50 / self.experiment.scale))
        self.orbit_stat.Record(self.Y)

        SimFunctions.Schedule(self.calendar, "Arrival", self.NSPP())
        SimFunctions.Schedule(self.calendar, "EndSimulation", self.experiment.run_length + self.experiment.warm_up)
        SimFunctions.Schedule(self.calendar, "ClearIt", self.experiment.warm_up)

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


        return [
            total_holding_cost / self.experiment.scale * 50,
            total_readmission_cost / self.experiment.scale * 50,
            total_intervention_cost / self.experiment.scale * 50,
            combined_cost / self.experiment.scale * 50,
            avg_intervention / self.experiment.scale * 50,
            self.wait_stat.Mean(),
            self.queue.Mean() / self.experiment.scale * 50,
            self.orbit_stat.Mean() / self.experiment.scale * 50,
            self.server.Mean() / self.experiment.scale * 50,
            self.X,
            self.Y,
            avg_congested_time / self.experiment.scale * 50,
            avg_intervention_time / self.experiment.scale * 50
        ]

def main():
    hs = [500, 1000, 1500, 2000, 2500, 3000]

    inits = [(10, 60), (60, 10), (60, 60)]
    
    #for x_0, y_0 in inits:
    #    for h in hs:
    for h in hs:
        for x_0, y_0 in inits:
            exp = Experiment(**{'scale': 50,
                                'N' : 45,
                                'mu': 1 / 5.6,
                                'nu': 1 / 12.07,
                                'r': 5000.0,
                                'x_0': x_0,
                                'y_0': y_0,
                                'warm_up': 0,
                                'run_length': 90,
                                'reps': 20000,
                                'h': h,
                                'M': 1110,
                                'lambda': 206.3/30*(1-p_h), # this is since lambda/(mu * (1-p_h)) = load * N
                                'fluctuation': 0.8,
                                'period': 1,
                                'stream': 12
                                })
            exp.run()
            exp.save()
        
main()
