import torch
import networkx as nx
import numpy as np
import os
from tqdm import tqdm 
import argparse
import h5py

class ShowerGenerator(object):
    def __init__(self, energy=100, *args, **kwargs) -> None:
        super().__init__()

        self.G = nx.DiGraph()

        self.G.add_node(0, energy=energy, position=np.array([0,0]), position_parent=np.array([0,0]), layer=0, active=True)

        self.delta_z = 2

        readout_layers = {0:list(range(5,9)), 1:list(range(9,18)), 2:list(range(18,28))}

        inv_map = {}
        for k, layers in readout_layers.items():
            for layer in layers:
                inv_map[layer] = k

        self.readout_layers = inv_map

        self.pc = [[] for i in range(len(readout_layers))]

        self.curr_layer = 0

    def propagate_and_decay_discrete(self, node, layer, read_out=None):
    
        energy = self.G.nodes[node]['energy']
        position = self.G.nodes[node]['position']
        position_parent = self.G.nodes[node]['position_parent']
        layer = self.G.nodes[node]['layer']
        active = self.G.nodes[node]['active']

        # sample if stops, decays, or passes
        p = np.random.uniform(0,1)

        p_stop = [1.0*(x/27)**0.75 for x in range(28)]

        # if stops
        if p < p_stop[layer] and layer > 6:
            # read-out
            self.G.nodes[node]['active'] = False

            if read_out is not None:
                self.pc[read_out].append(np.append(position, energy))

        # if decays
        #elif p < 1:
        elif (p < p_stop[layer] + 0.3 and layer > 6) or (p < 0.6 and layer <= 6):

            # sample opening angle and deviation angle
            theta = np.random.uniform(np.pi/16, np.pi/32)
            #theta = np.random.uniform(np.pi/24, np.pi/24)
            
            delta = np.random.uniform(-theta/4, theta/4)
            #delta = np.random.uniform(0, 0)

            # sample if direction is x or y
            direction = np.random.uniform(0,1)*np.pi*2

            theta1 = theta/2 - delta
            theta2 = theta/2 + delta

            # compute energies
            energy1 = energy*(theta2/theta)
            energy2 = energy*(theta1/theta)
            
            # this is assuming that (0,0,1) is aligned with (position - position_parent)
            delta_position_1 = self.rotate(np.array([self.delta_z*np.tan(theta1), 0]), direction)
            delta_position_2 = self.rotate(np.array([-self.delta_z*np.tan(theta2), 0]), direction)

            direction_1 = np.append(delta_position_1, self.delta_z)
            direction_2 = np.append(delta_position_2, self.delta_z)

            # recenter it
            rot_mat = self.center(np.append(position - position_parent, self.delta_z))
            direction_1 = rot_mat@direction_1
            direction_2 = rot_mat@direction_2

            assert direction_1[-1] > 0 and direction_2[-1] > 0

            position_1 = position + (direction_1*(self.delta_z/direction_1[-1]))[:2]
            position_2 = position + (direction_2*(self.delta_z/direction_2[-1]))[:2]

            # add nodes            
            num_nodes = self.G.number_of_nodes()            
            self.G.add_node(num_nodes, energy=energy1, position=position_1, position_parent=position, layer=layer+1, active=True)
            self.G.add_node(num_nodes+1, energy=energy2, position=position_2, position_parent=position, layer=layer+1, active=True)

        # if passes
        else:
            # pass through
            num_nodes = self.G.number_of_nodes()            
            self.G.add_node(num_nodes, energy=energy, position=position + (position - position_parent), position_parent=position, layer=layer+1, active=active)

    def rotate(self, v, theta):

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return R@v 

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

    def recenter(self, v, ref):
        M = self.center(ref)

        return M@v

    def generate(self):

        active_leaf_nodes = self.get_leaf_nodes(layer=self.curr_layer)

        read_out_layer = self.readout_layers.get(self.curr_layer, None)

        for node in active_leaf_nodes:
            self.propagate_and_decay_discrete(node, self.curr_layer, read_out_layer)

        self.curr_layer += 1

    def get_leaf_nodes(self, layer=0):
        return [x for x,y in self.G.nodes(data=True) if y['active']==True and y['layer']==layer]

    def read_out(self):
        pass

def pc2grid(all_showers_layer, bins):
    # all_showers_layer is a list of showers (single layer)

    shower_grids = []

    for shower in all_showers_layer:
        example = np.array(shower)
        H, xedges, yedges = np.histogram2d(example[:,0], example[:,1], weights=example[:,2], bins=bins)

        shower_grids.append(H)

    shower_grids = np.stack(shower_grids)

    return shower_grids

def get_parser():
    parser = argparse.ArgumentParser(
        description='Sythnetic Calorimeter Simulator with particle propagation and scattering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-N', '--num-showers', action='store', type=int, default=50,
                        help='Number of showers to generate.')

    parser.add_argument('-E', '--initial-energy', action='store', type=float, default=65.,
                        help='Initial energy of incoming particle (in GeV)')

    parser.add_argument('-grids', '--save-grids', action='store_true',
                        help='Whether to convert and save at grid granularity')

    parser.add_argument('-l', '--load', action='store_true',
                        help='No generation. Retrieve showers from save_path')

    parser.add_argument('-p', '--save-path', action='store', type=str,
                        help='Path where to save the generated showers')

    parser.add_argument('-s', '--seed', action='store', type=int, default=1234,
                        help='Numpy Random Seed')

    parser.add_argument('-bs', '--batch-size', action='store', type=int, default=10000,
                        help='Number of showers to store in a single file')

    

    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # Number of showers to generate
    N = int(args.num_showers)

    # Save Path
    save_path = args.save_path

    # Save grids
    save_grids = args.save_grids

    dir_path = os.path.abspath(os.path.join('gen', save_path))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    if args.load:
        pass

    else:
        np.random.seed(args.seed)
        batch_size = args.batch_size

        all_pcs = []
        energy = []

        for n in tqdm(range(N)):
            a = ShowerGenerator(energy=args.initial_energy)
            zero_hits = False

            for i in range(24):
                a.generate()
                
            num_layers = len(a.pc)

            # optionally remove events with zero hits in any layer
            for layer in range(num_layers):
                if len(a.pc[layer]) == 0:# or len(a.pc[1]) == 0 or len(a.pc[2]) == 0:
                    zero_hits = True

            if zero_hits:
                print('Skipped')
                continue

            
            all_pcs.append(a.pc)
            energy.append(np.array([args.initial_energy]))

            if (n+1)%batch_size == 0:   
                abs_save_path = os.path.abspath(os.path.join('gen', save_path, f'shower_batch{(n+1)//batch_size }.bin'))
                torch.save(all_pcs, abs_save_path)
                print(f'Saved file {abs_save_path}')

                abs_save_path = os.path.abspath(os.path.join('gen', save_path, f'tot_energy_batch{(n+1)//batch_size }.bin'))
                torch.save(energy, abs_save_path)

                all_pcs = []
                energy = []

        # save remaining
        if len(all_pcs) > 0:
            abs_save_path = os.path.abspath(os.path.join('gen', save_path, f'shower_batch{(n+1)//batch_size + 1}.bin'))
            torch.save(all_pcs, abs_save_path)
            print(f'Saved file {abs_save_path}')

            abs_save_path = os.path.abspath(os.path.join('gen', save_path, f'tot_energy_batch{(n+1)//batch_size + 1}.bin'))
            torch.save(energy, abs_save_path)


    scale = 1
    grids_config = [[np.linspace(-5,5,(4-1)*scale + 1), np.linspace(-5,5,(97-1)*scale + 1)], [np.linspace(-10,10,(13-1)*scale + 1), np.linspace(-10,10,(13-1)*scale + 1)], [np.linspace(-15,15,(13-1)*scale + 1), np.linspace(-15,15,(7-1)*scale + 1)]]
    
    #grids_config = [[np.linspace(-10,10,(13-1)*scale + 1), np.linspace(-10,10,(13-1)*scale + 1)]]


    if save_grids:

        img_arr_all_layers = {}
        E_arr_all_layers = []

        abs_save_path = os.path.abspath(os.path.join('gen', save_path, 'grids.hdf5'))
        f = h5py.File(abs_save_path, "w")

        abs_load_path = os.path.abspath(os.path.join('gen', save_path))
        shower_files = sorted([os.path.abspath(os.path.join('gen', save_path, x)) for x in os.listdir(abs_load_path) if 'shower' in x])
        energy_files = sorted([os.path.abspath(os.path.join('gen', save_path, x)) for x in os.listdir(abs_load_path) if 'energy' in x])

        assert len(shower_files) == len(energy_files)
        print(shower_files)

        for shower_file, energy_file in zip(shower_files, energy_files):
            print(f'Reading file {shower_file} ...')
            all_pcs = torch.load(shower_file)
            energy = torch.load(energy_file)

            layer_num=0
            for layer in range(len(all_pcs[0])):
                layer_num += 1
                all_showers_layer = [x[layer] for x in all_pcs]
                
                img_arr = pc2grid(all_showers_layer, grids_config[layer])

                if layer in img_arr_all_layers.keys():
                    img_arr_all_layers[layer].append(img_arr)
                else:
                    img_arr_all_layers[layer]  = [img_arr]

            
            E_arr_all_layers.append(energy)

        layer_num=0
        for layer in range(len(all_pcs[0])):
            layer_num += 1
            f.create_dataset(f'layer_{layer}', data=np.concatenate(img_arr_all_layers[layer], 0)*1000) # converting to MeV with *1000

        energies = np.concatenate(E_arr_all_layers, 0)
        f.create_dataset(f'energy', data=energies) 

        f.create_dataset(f'overflow', data=np.zeros((len(energies), 3)))

        f.close()

        print(f'Saved file {abs_save_path} ...')
