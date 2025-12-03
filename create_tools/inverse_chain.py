import numpy as np
import ase.io
import random
import re
from ase import Atom, Atoms
from ase.neighborlist import NeighborList
from scipy.spatial.transform import Rotation
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer


def calculate_inertia_tensor(atoms):
    """
    Calculate Inertia Tensor
    :param atoms: ASE Atoms 对象
    :return: 3x3 惯性张量矩阵
    """
    positions = atoms.get_positions()
    masses = atoms.get_masses()

    center_of_mass = atoms.get_center_of_mass()

    positions -= center_of_mass

    inertia_tensor = np.zeros((3, 3))

    for pos, mass in zip(positions, masses):
        x, y, z = pos
        inertia_tensor[0, 0] += mass * (y ** 2 + z ** 2)
        inertia_tensor[1, 1] += mass * (x ** 2 + z ** 2)
        inertia_tensor[2, 2] += mass * (x ** 2 + y ** 2)
        inertia_tensor[0, 1] -= mass * x * y
        inertia_tensor[0, 2] -= mass * x * z
        inertia_tensor[1, 2] -= mass * y * z

    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    return inertia_tensor


class InverseChainBuilder:
    def __init__(self, cluster_comp, slab, oxide_cif, m_num, o_num, dist):
        """
        Create inverse surface by cutting an atom chain from bulk
        :param cluster_comp:
        :param slab:
        :param oxide_cif:
        :param m_num:
        :param o_num:
        :param dist:
        """
        self.cluster_comp = cluster_comp
        self.cluster_m = self.get_metal()
        self.slab = slab
        self.cif = oxide_cif
        self.n_metal = m_num
        self.n_oxygen = o_num
        self.dis = dist
        self.bulk = ase.io.read(self.cif)
        self.mo_length = self.get_bond_length()
        self.cluster = None
        self.model = None

        self.metal_num_in_cif = self.bulk.get_chemical_symbols().count(self.cluster_m)
        self.repeat_times = int(np.ceil(np.cbrt(np.power(self.n_metal * 2, 3) / self.metal_num_in_cif)))
        self.bulk = self.bulk.repeat((self.repeat_times, self.repeat_times, self.repeat_times))
        self.bulk.pbc = [False, False, False]

    def build_cluster(self):
        atm = self.bulk.get_chemical_symbols()  # Atoms name
        pos = self.bulk.get_positions()  # Atomic coordinate

        cell = self.bulk.get_cell()
        center = cell.sum(axis=0) / 2

        m_indices = [i for i, atom in enumerate(self.bulk) if atom.symbol == self.cluster_m]
        distances = [np.linalg.norm(pos[i] - center) for i in m_indices]
        iat = m_indices[np.argmin(distances)]

        self.cluster = Atoms()
        self.cluster.append(Atom(atm[iat], pos[iat]))
        current_atom = iat
        current_label = self.bulk[iat].symbol
        existed_atoms = [iat]

        # Init search
        n_m = 1
        n_o = 0

        nbl = NeighborList([self.mo_length for _ in atm], self_interaction=False, bothways=True)
        nbl.update(self.bulk)

        while n_m < self.n_metal:
            ind, _ = nbl.get_neighbors(current_atom)
            if current_label == self.cluster_m:
                next_seq = [a for a in ind if self.bulk[a].symbol == "O"]
            else:
                next_seq = [a for a in ind if self.bulk[a].symbol == self.cluster_m]
            next_atom = random.sample(next_seq, k=1)[0]
            if next_atom not in existed_atoms:
                if self.bulk[next_atom].symbol == "O":
                    n_o += 1
                    current_label = "O"
                else:
                    n_m += 1
                    current_label = self.cluster_m
                self.cluster.append(Atom(atm[next_atom], pos[next_atom]))
                existed_atoms.append(next_atom)
                current_atom = next_atom

        if n_o < self.n_oxygen:
            neighbor_o = []
            for atom in existed_atoms:
                if self.bulk[atom].symbol == self.cluster_m:
                    ind, _ = nbl.get_neighbors(atom)
                    for a in ind:
                        if (a not in existed_atoms) and (a not in neighbor_o):
                            neighbor_o.append(a)
            extra_o = random.sample(neighbor_o, k=(self.n_oxygen - n_o))
            for atom in extra_o:
                self.cluster.append(Atom(atm[atom], pos[atom]))
        print("========== Cluster Created ==========")

    def place_cluster(self):
        cell_slab = self.slab.get_cell()
        atm_slab = self.slab.get_chemical_symbols()
        pos_slab = self.slab.get_positions()

        inertia_tensor = calculate_inertia_tensor(self.cluster)
        eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)  # Seek principal axis of inertia
        max_inertia_axis = eigenvectors[:, np.argmax(eigenvalues)]
        target_plane_normal = np.array([0, 0, 1])
        rotation = np.cross(max_inertia_axis, target_plane_normal)
        rotation_axis = rotation / np.linalg.norm(rotation)

        cos_theta = np.dot(max_inertia_axis, target_plane_normal)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        rotation_matrix = Rotation.from_rotvec(rotation_axis * theta).as_matrix()

        new_pos = np.dot(self.cluster.get_positions(), rotation_matrix.T)
        self.cluster.set_positions(new_pos)

        (x, y, _) = (cell_slab[0] + cell_slab[1]) / 2
        z_max = max(pos_slab[:, -1])

        natm_clus = self.cluster.get_global_number_of_atoms()
        atm_clus = self.cluster.get_chemical_symbols()
        pos_clus = self.cluster.get_positions()
        center_clus = [sum(pos_clus[:, dim])/natm_clus for dim in range(3)]
        center_clus[2] = min(pos_clus[:, -1])

        trans_vec = np.array([x, y, z_max]) - np.array(center_clus)
        trans_vec[2] += self.dis
        pos_clus_trans = np.array([pos + trans_vec for pos in pos_clus])

        atm = atm_slab + atm_clus
        pos = np.array(list(pos_slab) + list(pos_clus_trans))
        atoms = Atoms(atm, positions=pos, cell=cell_slab, pbc=True)
        self.model = atoms
        print("========== Cluster Placed ===========")

    def build_inverse(self):
        self.build_cluster()
        self.place_cluster()
        return self.model

    def get_bond_length(self):
        struc = AseAtomsAdaptor.get_structure(self.bulk)
        bv = BVAnalyzer()
        struc_oxi = bv.get_oxi_state_decorated_structure(struc)
        get_bond = CrystalNN()

        bond_length = []

        for i in range(len(struc)):
            neighs = get_bond.get_nn_info(struc_oxi, i)
            elem_i = struc[i].specie.symbol
            for nb in neighs:
                j = nb['site_index']
                elem_j = struc[j].specie.symbol
                d = struc.get_distance(i, j)
                if i < j and elem_i != elem_j:
                    bond_length.append(d)
        return max(bond_length) / 2

    def get_metal(self):
        elements = re.findall(r"[A-Z][a-z]?", self.cluster_comp)
        metals = [el for el in elements if el != "O"]
        if len(metals) != 1:
            raise ValueError("Invalid Oxide")
        return metals[0]
