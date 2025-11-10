from ase.build import bulk, surface, hcp0001
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator


class CleaveCreator:
    def __init__(self, atom_symbol, direction, xyz, vacuum, potential_path, job_type):
        """
        Cleave a pure metal surface, do supercell and add vacuum
        :param atom_symbol: slab atom type
        :param direction: a tuple, cleave direction e.g (1, 1, 1)
        :param xyz: a tuple, cell expansion factor on x,y,z axis
        :param vacuum: the thickness of vacuum
        """
        self.symbol = atom_symbol
        self.cleave_vec = direction
        self.expand = xyz
        self.thickness = vacuum
        self.potential = potential_path
        self.job = job_type
        self.structure = self.get_type_lattice()
        if self.structure['type'] == 'fcc' or self.structure['type'] == 'bcc':
            self.bulk = bulk(self.symbol, self.structure['type'], a=self.structure["a"])
        elif self.structure['type'] == 'hcp':
            self.bulk = bulk(self.symbol, self.structure['type'], a=self.structure["a"], c=self.structure["c"])
        else:
            raise ValueError("Invalid Atom Symbol or Not in Class Lib !")
        self.slab = None
        self.predictor = load_predict_unit(path=self.potential, device="cuda")
        self.calc = FAIRChemCalculator(self.predictor, task_name=self.job)  # Use omat for Inorganic Materials

    def lattice_optimize(self):
        self.bulk.calc = self.calc
        ucf = UnitCellFilter(self.bulk)
        print("========== Optimizing Bulk ==========")
        opt = LBFGS(ucf)
        opt.run(fmax=0.02)

    def supercell_vacuum(self):
        print("======= Surface Model Created =======")
        if self.structure['type'] == 'hcp' and self.cleave_vec == (0, 0, 1):
            self.slab = hcp0001(self.symbol, size=(self.expand[0], self.expand[1], self.expand[2]),
                                a=self.structure["a"], c=self.structure["c"], vacuum=self.thickness)
        else:
            self.slab = surface(self.bulk, self.cleave_vec, layers=self.expand[-1], vacuum=self.thickness)
            self.slab = self.slab.repeat((self.expand[0], self.expand[1], 1))

    def build_surface(self):
        self.lattice_optimize()
        self.supercell_vacuum()
        return self.slab

    def get_type_lattice(self):
        lattice_constant = {
            # fcc
            "Al": {"type": "fcc", "a": 4.05},
            "Cu": {"type": "fcc", "a": 3.61},
            "Ag": {"type": "fcc", "a": 4.09},
            "Au": {"type": "fcc", "a": 4.08},
            "Ni": {"type": "fcc", "a": 3.52},
            "Pt": {"type": "fcc", "a": 3.92},
            "Pb": {"type": "fcc", "a": 4.95},

            # bcc
            "Fe": {"type": "bcc", "a": 2.87},
            "Cr": {"type": "bcc", "a": 2.88},
            "Mo": {"type": "bcc", "a": 3.15},
            "W": {"type": "bcc", "a": 3.16},
            "V": {"type": "bcc", "a": 3.03},
            "Nb": {"type": "bcc", "a": 3.30},
            "Ta": {"type": "bcc", "a": 3.30},

            # hcp
            "Mg": {"type": "hcp", "a": 3.21, "c": 5.21},
            "Ti": {"type": "hcp", "a": 2.95, "c": 4.68},
            "Zn": {"type": "hcp", "a": 2.66, "c": 4.95},
            "Co": {"type": "hcp", "a": 2.51, "c": 4.07},
            "Zr": {"type": "hcp", "a": 3.23, "c": 5.15},
            "Hf": {"type": "hcp", "a": 3.19, "c": 5.05},
            "Be": {"type": "hcp", "a": 2.29, "c": 3.58},
            "Cd": {"type": "hcp", "a": 2.98, "c": 5.62},
        }
        return lattice_constant[self.symbol]
