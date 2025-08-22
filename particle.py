import numpy as np
from config import STEP_SIZE,BOX_SIZE

class Particle:
    def __init__(self,position=None):
        if position is None:
            self.position=np.random.uniform(-BOX_SIZE/2,BOX_SIZE/2,size=2)
        else:
            self.position=np.array(position,dtype=float)

    def step(self):
        angle=np.random.uniform(0,2*np.pi)
        dx=STEP_SIZE*np.cos(angle) 
        dy=STEP_SIZE*np.sin(angle)

        self.position +=np.array([dx,dy])
        self.position= np.clip(self.position,-BOX_SIZE/2,BOX_SIZE/2)

        return self.position.copy()
    