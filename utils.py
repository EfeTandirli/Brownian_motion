import numpy as np
from particle import Particle
from config import N_PARTICLES,STEPS

def simulate(n_particles=N_PARTICLES,steps=STEPS):
    particle=[Particle() for _ in range(n_particles)]
    traj= np.zeros((n_particles,steps,2))

    for t in range(steps):
        for i,p in enumerate(particle):
            traj[i,t]=p.step()

    return traj