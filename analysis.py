import numpy as np
import utils
def remove_drift(traj):
    mean_path=traj.mean(axis=0,keepdims=True)
    return traj -mean_path

def compute_msd(traj,max_lag=None):
    N,T,d = traj.shape
    if max_lag is None:
        max_lag= T-1
    msd= np.zeros(max_lag+1,dtype=float)

    lags=np.arange(max_lag+1)

    for tau in range(max_lag + 1):
        disp = traj[:, tau:, :] - traj[:, :T - tau, :]
        sq = np.sum(disp * disp, axis=2)           
        msd[tau] = sq.mean()
    return lags, msd

def estimate_diffusion_coeff(msd, dt, dim=2, fit_range=None):
    T = len(msd)
    t = np.arange(T) * dt
    if fit_range is None:
        start, end = max(1, T // 20), max(T // 2, 3)
    else:
        start, end = fit_range
    x = t[start:end].reshape(-1, 1)
    y = msd[start:end]
    A = np.hstack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    D = a / (2 * dim)
    return D, (a, b), (start, end)

def radial_distribution(traj, bins=50):
    r = np.linalg.norm(traj[:, -1, :], axis=1)
    hist, edges = np.histogram(r, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

def velocity_autocorrelation(traj, dt):
    vel = np.diff(traj, axis=1) / dt
    N, T, d = vel.shape
    vacf = np.zeros(T, dtype=float)
    for tau in range(T):
        prod = np.sum(vel[:, :T - tau, :] * vel[:, tau:, :], axis=2)
        vacf[tau] = prod.mean()
    lags = np.arange(T) * dt
    return lags, vacf

def radial_distribution(traj, bins=50):
    # distance from origin at final time
    r = np.linalg.norm(traj[:, -1, :], axis=1)
    hist, edges = np.histogram(r, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

def velocity_autocorrelation(traj, dt):
    vel = np.diff(traj, axis=1) / dt
    N, T, d = vel.shape
    vacf = np.zeros(T, dtype=float)
    for tau in range(T):
        prod = np.sum(vel[:, :T - tau, :] * vel[:, tau:, :], axis=2)
        vacf[tau] = prod.mean()
    lags = np.arange(T) * dt
    return lags, vacf