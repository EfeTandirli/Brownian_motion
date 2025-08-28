import numpy as np
from utils import simulate
import matplotlib.pyplot as plt
from analysis import remove_drift, compute_msd, estimate_diffusion_coeff


traj = simulate(n_particles=100, steps=2000)
traj_no_drift = remove_drift(traj)

lags, msd = compute_msd(traj_no_drift)

D, (a, b), (start, end) = estimate_diffusion_coeff(msd, dt=1.0)

print(f"Estimated diffusion coefficient: {D:.4f}")
plt.plot(lags, msd, label="MSD")
plt.plot(lags[start:end], a*lags[start:end] + b, '--', label=f"Fit (D={D:.4f})")
plt.xlabel("Lag time")
plt.ylabel("Mean Squared Displacement")
plt.legend()
plt.title("MSD and Diffusion Coefficient Estimate")
plt.show()