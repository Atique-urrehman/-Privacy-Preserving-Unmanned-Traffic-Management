import hashlib
import numpy as np
import matplotlib.pyplot as plt
import random

NUM_DRONES = 5
GRID_SIZE = 10
STEPS = 25

ledger = [] 

#hash function
def hash_position(x, y, z, t):
    data = f"{x}-{y}-{z}-{t}"
    return hashlib.sha256(data.encode()).hexdigest()

# drone class.
class Drone:
    def __init__(self, drone_id):
        self.id = drone_id

        #start pos
        self.position = np.array([
            random.uniform(0, GRID_SIZE),
            random.uniform(0, GRID_SIZE),
            random.uniform(0, GRID_SIZE / 2)
        ])

        #velocity
        self.velocity = np.array([
            random.uniform(-0.7, 0.7),
            random.uniform(-0.7, 0.7),
            random.uniform(-0.3, 0.3)
        ])

        self.path = []
        self.hashes = []

    def move(self, t):
        self.position += self.velocity

        #boundary handling /bounce back
        for i in range(3):
            if self.position[i] < 0 or self.position[i] > GRID_SIZE:
                self.velocity[i] *= -1
                self.position[i] = max(0, min(self.position[i], GRID_SIZE))

        x, y, z = self.position

        #discretize into grid cell
        cell = (int(x), int(y), int(z), t)
        self.path.append(cell)

        #hash
        h = hash_position(*cell)
        self.hashes.append(h)

        #store in ledger
        ledger.append({
            "drone": self.id,
            "time": t,
            "position": cell,
            "hash": h
        })

        return x, y, z


#detect collision
def detect_collisions(drones):
    collisions = []

    for i in range(len(drones)):
        for j in range(i + 1, len(drones)):
            common = set(drones[i].hashes).intersection(set(drones[j].hashes))

            if common:
                collisions.append((drones[i].id, drones[j].id, list(common)[0]))

    return collisions


#print ledger
def print_latest_entries():
    print("\n   --- Latest Ledger Entries ")
    for entry in ledger[-NUM_DRONES:]:
        print(f"Drone {entry['drone']} | Time {entry['time']} | "
              f"Pos {entry['position']} | Hash {entry['hash'][:10]}")


def simulate():
    #create multiple drones dynamically
    drones = [Drone(i+1) for i in range(NUM_DRONES)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for t in range(STEPS):
        ax.clear()

        #grid setup
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_zlim(0, GRID_SIZE / 2)

        ax.set_title(f"3D Drone Simulation | Time {t}")

        positions = []

        # move all drones
        for drone in drones:
            x, y, z = drone.move(t)
            positions.append((x, y, z))

        print_latest_entries() #hash print

        #plot all drones and paths
        for drone in drones:
            path = np.array([(p[0], p[1], p[2]) for p in drone.path])

            if len(path) > 0:
                ax.plot(path[:,0], path[:,1], path[:,2], linestyle='dashed')

            #current position
            x, y, z = drone.position
            ax.scatter(x, y, z)

        #detect collisions
        collisions = detect_collisions(drones)

        if collisions:
            for d1, d2, h in collisions:
                print(f"\n Collision between Drone {d1} & Drone {d2}")
                print(f"Hash: {h[:12]}")
            ax.set_title(" Collision Detected!")

        plt.pause(0.4)

    plt.show()


simulate()