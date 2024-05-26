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

# Change to the directory you want the ouput to be saved in
directory = "quad_trans_alt\\"
log_file = "sims.log"


T_MAX = 365
MAX_STEP = 0.02


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

        self.interv_cost_str = kwargs.get('C', "lambda p: 10 * 1.0 * (0.2-p)")
        self.interv_cost = eval(self.interv_cost_str)

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

        # compute the ingredients needed for the optimal policy
        self.taus = []
        policies = []

        tau_step = kwargs.get('tau_steps', 1e-1)
        tau = 0
        p = self.p_eq
        while p > self.p_l + 1e-4:
            tau += tau_step
            p = self.p(tau)
            self.taus.append(tau)
            policies.append(p)
        self.policies = np.array(policies)
        self.constants = np.array([self.const(tau) for tau in self.taus])
        self.y_coefs = np.array([self.y_coef(tau) for tau in self.taus])


        # DO THE KNN for the unknown area

        y0s = np.linspace(0, (self.service_rate * self.server_num - self.arrival_rate) / self.return_rate, 300)
        sols = np.vectorize(self.solve_trajectory)(y0s)
        states_list = []
        ps_list = []

        for sol in sols:
            xs = sol.y[0]
            ys = sol.y[1]
            cutoff = len(xs)
            if min(xs) < 0:
                cutoff = min((xs < 0).argmax(), cutoff)
            if min(ys) < 0:
                cutoff = min((ys < 0).argmax(), cutoff)
            ps = np.vectorize(self.pol)(sol.y[-1][:cutoff])

            states_list.append(np.column_stack((xs[:cutoff], ys[:cutoff])))
            ps_list.append(ps)

        all_states = np.concatenate(states_list)
        all_ps = np.concatenate(ps_list)
        self.knn = KNeighborsRegressor(n_neighbors=1)
        self.knn.fit(all_states, all_ps)


        self.policy_variants = [self.eq_policy, self.simple_policy, self.optimal_policy]


    def run(self):
        for j, pol in enumerate(self.policy_variants):
            # SimRNG.InitializeRNSeed()
            for i in range(self.reps):
                sim = Simulation(self, policy=pol)
                results = sim.simulate()#stream=i)
                self.metrics[j].append(list(results))

    def save(self):
        timestamp = datetime.now()
        time_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        path = directory + "simulation_" + time_str + ".csv"
        headers = ['h_cost', 'r_cost', 'C_cost', 'total_cost', 'mean_p', 'mean_wait', 'mean_queue', 'mean_orbit', 'mean_server', 'x_T', 'y_T', 'mean_congested_time', 'mean_p_time']
        eq_frame = pd.DataFrame(self.metrics[0], columns=headers)
        simple_frame = pd.DataFrame(self.metrics[1], columns=headers)
        opt_frame = pd.DataFrame(self.metrics[2], columns=headers)
        frame = pd.concat([eq_frame, simple_frame, opt_frame], axis=1)
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

        # AllWaitMean.append(self.wait_stat.Mean())
        # AllOrbitMean.append(self.orbit_stat.Mean())
        # AllQueueMean.append(self.queue.Mean())
        # AllQueueNum.append(self.queue.NumQueue())
        # AllServerBusyMean.append(self.server.Mean())


def main():
    nus = [1/10, 1/15, 1/20]
    rates = [9.0, 9.5, 9.8]
    
    inits = [(25, 25), (25, 65), (65, 25), (65, 65)]

    for nu, rate, init in product(nus, rates, inits):

        exp = Experiment(**{'scale': 50,
                            'lambda': rate,
                            'nu': nu,
                            'mu': 1 / 4,
                            'h': 0.2,
                            'C': "lambda p: 100*0.5*(0.2-p)**2",
                            'x_0': init[0],
                            'y_0': init[1],
                            'reps': 2000,
                            'fluctuation': 0,
                            'period': 30,
                            'warm_up': 0,
                            'run_length': 90,
                            })
        exp.run()
        exp.save()

main()
