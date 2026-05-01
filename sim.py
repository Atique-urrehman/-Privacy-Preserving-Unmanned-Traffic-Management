import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class VoxelDiscretizer:
    
    def __init__(self, voxel_size=1.0):
        self.voxel_size = voxel_size
    
    def get_voxel(self, x, y, z):
        vx = int(np.floor(x / self.voxel_size))
        vy = int(np.floor(y / self.voxel_size))
        vz = int(np.floor(z / self.voxel_size))
        return (vx, vy, vz)
    
    def get_voxel_center(self, vx, vy, vz):
        x = (vx + 0.5) * self.voxel_size
        y = (vy + 0.5) * self.voxel_size
        z = (vz + 0.5) * self.voxel_size
        return (x, y, z)


class Drone:
    
    def __init__(self, drone_id, start_pos, trajectory_func, color, name):
        
        self.drone_id = drone_id
        self.start_pos = np.array(start_pos, dtype=float)
        self.position = np.array(start_pos, dtype=float)
        self.trajectory_func = trajectory_func
        self.color = color
        self.name = name
        self.history = [np.array(start_pos, dtype=float)]
        self.is_halted = False
        self.collision_time = None
        self.resume_time_step = None

    def reset(self):
        self.position = np.array(self.start_pos, dtype=float)
        self.history = [np.array(self.start_pos, dtype=float)]
        self.is_halted = False
        self.collision_time = None
        self.resume_time_step = None
    
    def update(self, time_step, discretizer):
        if not self.is_halted:
            new_pos = self.trajectory_func(time_step)
            self.position = np.array(new_pos, dtype=float)
            self.history.append(np.array(new_pos))
    
    def get_voxel(self, discretizer):
        return discretizer.get_voxel(self.position[0], self.position[1], self.position[2])
    
    def halt(self, time_step):
        self.is_halted = True
        self.collision_time = time_step
        self.resume_time_step = None

    def pause_until(self, resume_time_step, time_step):
        self.is_halted = True
        self.collision_time = time_step
        self.resume_time_step = resume_time_step

    def maybe_resume(self, time_step):
        if self.is_halted and self.resume_time_step is not None and time_step >= self.resume_time_step:
            self.is_halted = False
            self.resume_time_step = None


class CollisionManager:
    
    def __init__(self):
        self.epoch_window = 5
        self.current_epoch = None
        self.occupancy_log = {}  
        self.collisions = []
    
    def get_epoch(self, time_step):
        return time_step // self.epoch_window

    def _generate_hash(self, voxel, epoch):
        vx, vy, vz = voxel
        data_string = f"{vx}:{vy}:{vz}:{epoch}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def register_occupancy(self, voxel, time_step, drone_id):
        epoch = self.get_epoch(time_step)
        
        if self.current_epoch is not None and epoch != self.current_epoch:
            self.occupancy_log.clear()
        self.current_epoch = epoch
        
        commitment_hash = self._generate_hash(voxel, epoch)
        
        if commitment_hash in self.occupancy_log:
            existing_entry = self.occupancy_log[commitment_hash]
            first_drone_id = existing_entry[2]
            
            if first_drone_id != drone_id:
                self.collisions.append((time_step, first_drone_id, drone_id, voxel, commitment_hash, epoch))
                return False
            else:
                return True
        
        self.occupancy_log[commitment_hash] = (voxel, time_step, drone_id)
        return True
    
    def get_collision_location(self):
        if self.collisions:
            return self.collisions[0][3]
        return None


def drone_a_trajectory(t):
    x = 1 + 7 * (t / 15.0)
    y = 1 + 7 * (t / 15.0)
    z = 1 + 7 * (t / 15.0)
    return (x, y, z)


def drone_b_trajectory(t):
    x = 8 - 7 * (t / 20.0)
    y = 8 - 7 * (t / 20.0)
    z = 1 + 7 * (t / 20.0)
    return (x, y, z)


def drone_c_trajectory(t):
    if t < 3:
        return (1, 8, 0.5)
    
    adjusted_t = t - 3
    x = 1 + 7 * (adjusted_t / 18.0)
    y = 8 - 7 * (adjusted_t / 18.0)
    z = 0.5 + 7 * (adjusted_t / 18.0)
    return (x, y, z)


class DroneSimulation:
    
    def __init__(self, total_time_steps=20):
        self.time_step = 0
        self.last_frame = None
        self.total_time_steps = total_time_steps
        self.discretizer = VoxelDiscretizer(voxel_size=1.0)
        self.collision_manager = CollisionManager()
        
        self.drones = [
            Drone(0, (1, 1, 1), drone_a_trajectory, 'blue', 'Drone A'),
            Drone(1, (8, 8, 1), drone_b_trajectory, 'red', 'Drone B'),
            Drone(2, (1, 8, 0.5), drone_c_trajectory, 'green', 'Drone C'),
        ]
        
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.scatter_plots = [
            self.ax.scatter([], [], [], s=100, c=[d.color], 
                          alpha=0.8, label=d.name, edgecolors='black', linewidth=1.5)
            for d in self.drones
        ]
        
        self.trail_plots = [
            self.ax.plot([], [], [], color=d.color, alpha=0.3, linewidth=1.0)[0]
            for d in self.drones
        ]
        
        self.collision_marker = self.ax.scatter([], [], [], s=400, marker='x', 
                                               c='red', alpha=1.0, linewidths=3)
        
        self.setup_axes()

    def reset_state(self):
        self.time_step = 0
        self.collision_manager = CollisionManager()
        for drone in self.drones:
            drone.reset()
        self.collision_marker._offsets3d = ([], [], [])
        self.info_text.set_text('')
    
    def setup_axes(self):
        self.ax.set_xlim(-1, 10)
        self.ax.set_ylim(-1, 10)
        self.ax.set_zlim(-1, 10)
        self.ax.set_xlabel('X (voxels)', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Y (voxels)', fontsize=10, fontweight='bold')
        self.ax.set_zlabel('Z (voxels)', fontsize=10, fontweight='bold')
        self.ax.set_title('Blind Air Traffic Control - 3D Drone Simulation', 
                         fontsize=14, fontweight='bold')
        
        self.draw_voxel_grid()
        
        self.ax.legend(loc='upper left', fontsize=10)
        
        self.info_text = self.fig.text(0.02, 0.02, '', fontsize=10, 
                                       family='monospace', verticalalignment='bottom')
    
    def draw_voxel_grid(self):
        alpha = 0.1
        color = 'gray'
        
        for i in range(0, 10):
            for j in range(0, 10):
                self.ax.plot([i, i], [j, j], [0, 9], color=color, alpha=alpha, linewidth=0.5)
                self.ax.plot([i, i], [0, 9], [j, j], color=color, alpha=alpha, linewidth=0.5)
                self.ax.plot([0, 9], [i, i], [j, j], color=color, alpha=alpha, linewidth=0.5)
    
    def update_frame(self, frame):
        if self.last_frame is not None and frame < self.last_frame:
            self.reset_state()

        self.time_step = frame
        self.last_frame = frame

        for drone in self.drones:
            drone.maybe_resume(frame)
        
        for drone in self.drones:
            drone.update(frame, self.discretizer)
        
        for drone in self.drones:
            if not drone.is_halted:
                voxel = drone.get_voxel(self.discretizer)
                epoch = self.collision_manager.get_epoch(frame)
                is_safe = self.collision_manager.register_occupancy(voxel, frame, drone.drone_id)
                
                if not is_safe:

                    resume_time_step = (epoch + 1) * self.collision_manager.epoch_window
                    drone.pause_until(resume_time_step, frame)
                    collision_info = self.collision_manager.collisions[-1]
                    time_step_val = collision_info[0]
                    first_drone_id = collision_info[1]
                    conflicting_drone_id = collision_info[2]
                    voxel_coords = collision_info[3]
                    collision_hash = collision_info[4]
                    epoch_bucket = collision_info[5]
                    
                    print(f"COLLISION DETECTED!")
                    print(f"Time Step: {time_step_val} (Epoch Bucket: {epoch_bucket})")
                    print(f"Location (voxel): {voxel_coords}")
                    print(f"Hash Commitment: {collision_hash[:16]}...")
                    print(f"First Drone (Reserved): {self.drones[first_drone_id].name} (ID: {first_drone_id})")
                    print(f"Conflicting Drone: {self.drones[conflicting_drone_id].name} (ID: {conflicting_drone_id})")
                    print(f"Arbitration: {self.drones[conflicting_drone_id].name} HALTED (First-Come-First-Served)")
        
        for i, drone in enumerate(self.drones):
            scatter = self.scatter_plots[i]
            xs = [drone.position[0]]
            ys = [drone.position[1]]
            zs = [drone.position[2]]
            scatter._offsets3d = (xs, ys, zs)
        
        for i, drone in enumerate(self.drones):
            if len(drone.history) > 1:
                history = np.array(drone.history)
                self.trail_plots[i].set_data(history[:, 0], history[:, 1])
                self.trail_plots[i].set_3d_properties(history[:, 2])
        
        collision_voxel = self.collision_manager.get_collision_location()
        if collision_voxel:
            cx, cy, cz = self.discretizer.get_voxel_center(*collision_voxel)
            self.collision_marker._offsets3d = ([cx], [cy], [cz])
        
        halted_drones = [d for d in self.drones if d.is_halted]
        halted_info = f"Halted: {', '.join([d.name for d in halted_drones])}" if halted_drones else "Status: Normal operations"
        
        info = f"Time Step: {frame}/{self.total_time_steps}\n"
        info += f"Active Drones: {sum(1 for d in self.drones if not d.is_halted)}/3\n"
        info += f"{halted_info}\n"
        info += f"Collisions Detected: {len(self.collision_manager.collisions)}"
        
        self.info_text.set_text(info)
        
        return self.scatter_plots + self.trail_plots + [self.collision_marker, self.info_text]
    
    def run(self):
        ani = FuncAnimation(self.fig, self.update_frame, frames=self.total_time_steps,
                          interval=500, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    print("BLIND AIR TRAFFIC CONTROL SYSTEM INITIALIZATION")
    print("\nSimulation Parameters:")
    print("  Airspace: 10 x 10 x 10 voxels")
    print("  Voxel Size: 1.0 unit³")
    print("  Total Time Steps: 20")
    print("  Drones: 3")
    print("\nDrone Routes:")
    print("  Drone A (Blue): Diagonal path (1,1,1) → (8,8,8) - NO COLLISION")
    print("  Drone B (Red): Collision course - WILL COLLIDE with A around t=11-13")
    print("  Drone C (Green): Safe delayed path - Passes AFTER collision clears")
    print("\nListen for collision alerts in the console during simulation...")
    
    sim = DroneSimulation(total_time_steps=20)
    sim.run()
