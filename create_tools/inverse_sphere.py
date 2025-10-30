import numpy as np
import ase.io
from ase import Atom, Atoms
from ase.neighborlist import NeighborList


class InverseSphereBuilder:
    def __init__(self, slab_metal, cluster_metal, slab, oxide_cif, rcut, dist):
        """
        Create inverse surface using hemi-sphere cut
        :param slab: slab model, expecting atoms type from ase
        :param oxide_cif: path of cif model, primitive cell recommended
        :param rcut: hemi-sphere radius
        :param dist: cluster's height above slab
        """
        self.slab_m = slab_metal
        self.cluster_m = cluster_metal
        self.slab = slab
        self.cif = oxide_cif
        self.cut = rcut
        self.dis = dist
        self.bulk = ase.io.read(self.cif)
        self.cluster = None
        self.model = None

    def build_cluster(self, iat=0):
        n_atm = self.bulk.get_global_number_of_atoms()
        cell = self.bulk.get_cell()
        atm = self.bulk.get_chemical_symbols()
        pos = self.bulk.get_positions()

        nbl = NeighborList([self.cut/2] * n_atm, skin=0.0, self_interaction=False, bothways=True)
        nbl.update(self.bulk)
        ind, off = nbl.get_neighbors(iat)

        self.cluster = []
        self.cluster.append(Atom(atm[iat], pos[iat]))
        for i in range(len(ind)):
            _iat = ind[i]
            _atm = atm[_iat]
            _pos = pos[_iat] + np.dot(off[i], cell)
            if _pos[2] >= pos[iat][2]:
                self.cluster.append(Atom(_atm, _pos))

        self.cluster = Atoms(self.cluster)
        print("========== Cluster Created ==========")

    def place_cluster(self):
        cell_slab = self.slab.get_cell()
        atm_slab = self.slab.get_chemical_symbols()
        pos_slab = self.slab.get_positions()

        (x, y, _) = (cell_slab[0] + cell_slab[1]) / 2
        z_max = max(pos_slab[:, -1])

        n_atm_clus = self.cluster.get_global_number_of_atoms()
        atm_clus = self.cluster.get_chemical_symbols()
        pos_clus = self.cluster.get_positions()

        center_clus = [sum(pos_clus[:, i]) / n_atm_clus for i in range(3)]
        center_clus[2] = min(pos_clus[:, -1])

        trans_vec = np.array([x, y, z_max]) - np.array(center_clus)
        trans_vec[2] += self.dis
        pos_clus_trans = np.array([pos + trans_vec for pos in pos_clus])

        atm = atm_slab + atm_clus
        pos = list(list(pos_slab) + list(pos_clus_trans))
        self.model = Atoms(atm, positions=pos, cell=cell_slab, pbc=True)
        print("========== Cluster Placed ===========")

    def build_inverse(self, iat=0):
        self.build_cluster(iat=iat)
        self.place_cluster()
        return self.model
