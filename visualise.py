import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_traj(traj,interval=50,save_path=None):
    fig, ax= plt.subplots()
    scat= ax.scatter([],[],s=20)

    all_x= traj[:,:,0].flatten()
    all_y = traj[:,:,1].flatten()
    margin=5

    ax.set_xlim(all_x.min()-margin, all_x.max()+margin)
    ax.set_ylim(all_y.min()-margin, all_y.max()+margin)
    ax.set_aspect("equal")
    ax.set_title("Brownian Motion")

    def update(frame):
        scat.set_offsets(traj[:,frame,:])
        return scat
    
    anim=FuncAnimation(fig,update,frames=traj.shape[1],interval=interval)

    if save_path:
        anim.save(save_path)
    else:
        plt.show()