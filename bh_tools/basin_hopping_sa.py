from ase import Atoms
from ase.optimize.basin import BasinHopping
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from bh_tools.bh_early_stop import BasinHoppingEarlyStop
import numpy as np


class BasinHoppingSA:
    def __init__(self, system: Atoms, t_beg: float, t_end: float, hop_steps: int, n_stages: int,
                 disturbance: float, f_max: float, step_max: int,
                 out_path: str, log_file_path: str, potential_path: str, job_type: str):
        self.atoms = system
        self.t_beg = t_beg
        self.t_end = t_end
        self.hopping_step = hop_steps
        self.stages = n_stages
        self.scale = disturbance
        self.ediff = f_max
        self.step_max = step_max
        self.out_path = out_path
        self.log_path = log_file_path
        self.potential = potential_path
        self.job = job_type
        self.optimizer = LBFGS
        self.model = load_predict_unit(path=self.potential, device="cuda")
        self.calculator = FAIRChemCalculator(self.model, task_name=self.job)
        self.atoms.calc = self.calculator
        self.best_energy = float('inf')
        self.best_struc = None
        # Clear log file
        with open(self.log_path, 'w'):
            pass

    def run_basin_hopping_sa(self):
        anneal_schedule = np.linspace(self.t_beg, self.t_end, self.stages)

        for t in anneal_schedule:
            bh = BasinHoppingEarlyStop(self.atoms,
                                       temperature=t,
                                       logfile=self.log_path,
                                       optimizer=self.optimizer,
                                       fmax=self.ediff,
                                       dr=self.scale,
                                       step_max=self.step_max,
                                       out_path=self.out_path)
            bh.run(steps=self.hopping_step)
            _, self.best_struc = bh.get_minimum()
            self.atoms.positions = self.best_struc.positions
        return self.best_struc


if __name__ == '__main__':
    from ase.visualize import view
    from create_tools.cleave_vacuum import CleaveCreator
    from create_tools.inverse_chain import InverseChainBuilder
    from constrain_tools.constrain_slab import ConstrainLayers

    cc = CleaveCreator(atom_symbol="Ni", direction=(1, 1, 1), xyz=(6, 6, 3),
                       vacuum=15, job_type="omat", potential_path="D:/fairchem_models/uma-s-1p1.pt")
    Slab = cc.build_surface()
    icb = InverseChainBuilder(cluster_comp="TcO2", slab=Slab, m_num=3, o_num=6,
                              oxide_cif='../oxide_models/TcO2.cif', dist=2)
    Model = icb.build_inverse()
    cl = ConstrainLayers(Model, sub_layers=3)
    Model = cl.set_constrain()
    view(Model)
    bh_sa = BasinHoppingSA(system=Model, t_beg=1000, t_end=500, hop_steps=20, n_stages=6,
                           disturbance=0.3, f_max=0.10, step_max=200, log_file_path="../Test/bh_sa.log",
                           potential_path="D:/fairchem_models/uma-s-1p1.pt", job_type="omat", out_path='../Test/')
    opted_struc = bh_sa.run_basin_hopping_sa()
    view(opted_struc)
