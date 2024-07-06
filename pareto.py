# this python file describes a class called pareto,
# which is used to generate pareto frontiers via epsilon constraint method
# with working with calliope model

import pandas as pd
import calliope


# Steps to achieve multi-objective optimization (ε-cut):
# 1. prepare cost data (monetary: HSLU database; emission: KBOB, limitation: different database);
# 2. Input into calliope configuration;
# 3. Define available technology;
# 4. Solve for min-cost (C_L, E_L) and min-emission (C_R, E_R);
# 5. define amount of cuts (n), and primary objective (normally Cost)
# 6. Divide emission range [E_L, E_R] into n parts, E_0 = E_L, E_1, ..., E_i, ..., E_n-1, E_n=E_R
# 7. optimize for C, with constriaint of E≤E_i
# 8. get n+1 points: (C_0, E_0) = (C_L, E_L), (C_1, E_1), ..., (C_i, E_i), ..., (C_n-1, E_n-1), (C_n, E_n) = (C_R, E_R)
# 9. link these points in a coordinate plane to form the pareto front.
class Pareto:
    def __init__(self, model, config_dict):
        self.model = model
        self.config_dict = config_dict

    def