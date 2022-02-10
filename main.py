import torch
import networkx as nx
import numpy as np
import os, sys
from tqdm import tqdm 

class ShowerGenerator(object):
    def __init__(self, mass, velocity, *args, **kwargs) -> None:
        super().__init__()

        self.G = nx.DiGraph()

        # add initial particle
        self.G.add_node(0, mass=mass, position=np.array([0,0,0]), velocity=velocity, timestamp=0)

        self.pc = []

    def propagate_and_decay(self, node):

        # print(f'Node = {node}')

        mass = self.G.nodes[node]['mass']
        timestamp = self.G.nodes[node]['timestamp']
        position = self.G.nodes[node]['position']
        velocity = self.G.nodes[node]['velocity']

        # print(f'Position : {position}')

        # Sample distance or time before decay happens
        deltaT = 2 # seconds
        position_new = position + deltaT*velocity

        # Readout at some z = Z
        if (self.layer_z - position[-1])*(self.layer_z - position_new[-1]) < 0:
            interpolation_k = (self.layer_z - position[-1])/(position_new[-1] - position[-1])
            hit_coord = position + interpolation_k*(position_new - position)
            hit_energy = 0.5*mass*np.linalg.norm(velocity)
            self.pc.append(np.append(hit_coord[:2], hit_energy))
    
        # Sample number of splits
        num_splits = 2 # prob is dirac_delta(N=2)

        # Sample masses
        eps = mass/4
        mass_1 = np.random.uniform(mass/2 + eps, mass/2 -eps)
        mass_2 = mass - mass_1
        # print(f'Sampled Masses = {mass_1} and {mass_2}')

        # Sample velocities
        # print(f'Velocity = {velocity}')

        ## Recenter if needed # (0, 0, v_z)
        rot_matrix = self.center(velocity)
        velocity_rot = rot_matrix@velocity
        # print(f'Velocity Rotated = {velocity_rot}')

        velocity_z = velocity_rot[-1]
        
        velocity_1x = np.random.uniform(-0.05*velocity_z, 0.05*velocity_z)
        velocity_2x = -(mass_1/mass_2)*velocity_1x

        velocity_1y = np.random.uniform(-0.05*velocity_z, 0.05*velocity_z)
        velocity_2y = -(mass_1/mass_2)*velocity_1y

        eps = 0.1*velocity_z
        velocity_1z = np.random.uniform(velocity_z-eps, velocity_z+eps)
        velocity_2z = (mass*velocity[-1] - mass_1*velocity_1z)/mass_2

        velocity_1 = np.array([velocity_1x, velocity_1y, velocity_1z])
        velocity_2 = np.array([velocity_2x, velocity_2y, velocity_2z])

        # print(f'Sample Velocity 1  = {velocity_1}')
        # print(f'Sample Velocity 2  = {velocity_2}')

        # Recenter velocity_1 and velocity_2 if needed
        velocity_1 = rot_matrix.T@velocity_1
        velocity_2 = rot_matrix.T@velocity_2

        # print(f'Sample Velocity 1 Rotated = {velocity_1}')
        # print(f'Sample Velocity 2 Rotated = {velocity_2}')

        # append both nodes to graph
        num_nodes = self.G.number_of_nodes()
        self.G.add_node(num_nodes, mass=mass_1, velocity=velocity_1, timestamp=timestamp+deltaT, position=position_new)
        self.G.add_node(num_nodes+1, mass=mass_2, velocity=velocity_2, timestamp=timestamp+deltaT, position=position_new)
        
        self.G.add_edge(node, num_nodes)
        self.G.add_edge(node, num_nodes + 1)

    def center(self, v):
        m3= v/np.linalg.norm(v)
        m2 = np.zeros(3)
        m2[1] = -m3[2]
        m2[2] = m3[1]
        m2 = m2/np.linalg.norm(m2)
        m1 = np.cross(m2,m3)
        M = np.array([m1,m2,m3])
        # Multiply with M to go from v to (0,0,1)
        # Multiply with M.T to go from (0,0,1) to v
        return M


    def generate(self):

        leaf_nodes = self.get_leaf_nodes()

        for node in leaf_nodes:
            self.propagate_and_decay(node)

    def get_leaf_nodes(self):
        return [key for key, val in self.G.out_degree if val==0]

    def read_out(self):
        pass

if __name__ == "__main__":

    # Number of showers to generate
    N = int(sys.argv[1])

    # Save Path
    save_path = sys.argv[2]

    all_pcs = []

    for n in tqdm(range(N)):

        a = ShowerGenerator(mass=10, velocity=np.array([0,0,10]))

        for i in range(12):
            a.generate()

        all_pcs.append(a.pc)

    
    abs_save_path = os.path.abspath(os.path.join('gen',save_path))
    torch.save(all_pcs, abs_save_path)
    print(f'Saved file {abs_save_path}')
