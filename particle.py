import numpy as np
from config import STEP_SIZE,BOX_SIZE,DRIFT



class Particle:
    def __init__(self,position=None,mode=None):
        if position is None and mode is 1:
            self.position=np.random.uniform(-BOX_SIZE/2,BOX_SIZE/2,size=2)
        elif position is None and mode is 0:
            std=0.01*BOX_SIZE
            self.position=np.random.normal(0,std,size=2)
        else:
            self.position=np.array(position,dtype=float)

    def step(self):
        angle=np.random.uniform(0,2*np.pi)
        dx=STEP_SIZE*np.cos(angle) 
        dy=STEP_SIZE*np.sin(angle)

        self.position +=np.array([dx,dy])+ DRIFT
        self.position= np.clip(self.position,-BOX_SIZE/2,BOX_SIZE/2)

        return self.position.copy()
    