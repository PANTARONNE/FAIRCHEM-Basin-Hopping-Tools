from typing import IO, Type, Union
import numpy as np
from copy import deepcopy
from ase import Atoms, units, neighborlist
from ase.io.trajectory import Trajectory
from ase.optimize.fire import FIRE
from ase.optimize.optimize import Dynamics, Optimizer
from ase.optimize.basin import BasinHopping
from ase.parallel import world
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation


class BasinHoppingEarlyStop(BasinHopping):
    def __init__(
        self,
        atoms: Atoms,
        out_path: str,
        temperature: float = 100 * units.kB,
        optimizer: Type[Optimizer] = FIRE,
        fmax: float = 0.1,
        step_max: int = 200,
        dr: float = 0.1,
        ratio: float = 0.8,
        cutoff_factor: float = 1.2,
        logfile: Union[IO, str] = '-',
        optimizer_logfile: str = '-',
        adjust_cm: bool = True,
    ):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        """
        self.out_path = out_path
        self.kT = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.step_max = step_max
        self.dr = dr
        self.ratio = ratio
        self.cutoff_factor = cutoff_factor
        self.logfile = logfile
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        self.optimizer_logfile = optimizer_logfile
        self.trajectory = f'%slowest.traj' % self.out_path
        self.lm_trajectory = f"%slocal_minima.traj" % self.out_path

        if isinstance(self.lm_trajectory, str):
            self.lm_trajectory = self.closelater(
                Trajectory(self.lm_trajectory, 'w', atoms))

        Dynamics.__init__(self, atoms, self.logfile, self.trajectory)
        self.initialize()

    def is_connected(self):
        atoms = self._atoms()
        cutoffs = neighborlist.natural_cutoffs(atoms, mult=self.cutoff_factor)
        nl = neighborlist.NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        matrix = nl.get_connectivity_matrix(sparse=True)
        n_components, _ = connected_components(matrix)
        return n_components == 1

    def get_energy(self, positions):
        """Return the energy of the nearest local minimum."""
        if np.any(self.positions != positions):
            self.positions = positions
            self.optimizable.set_x(positions.ravel())

            with self.optimizer(self.optimizable,
                                logfile=self.optimizer_logfile) as opt:
                opt.run(fmax=self.fmax, steps=self.step_max)

            if self.optimizable.get_gradient().max() > self.fmax:  # When the optimization does not converge
                self.energy = float('inf')
            elif not self.is_connected():
                self.energy = float('inf')
            else:
                self.energy = self.optimizable.get_value()
                if self.lm_trajectory is not None:
                    self.lm_trajectory.write(self.optimizable)
        return self.energy

    def move(self, ro):
        """Move atoms by a random step."""
        atoms = self._atoms()
        # displace coordinates
        if np.random.uniform() < self.ratio:
            disp = np.random.uniform(-1., 1., (len(atoms), 3))
            rn = ro + self.dr * disp
            atoms.set_positions(rn)
            if self.cm is not None:
                cm = atoms.get_center_of_mass()
                atoms.translate(self.cm - cm)
        else:
            theta = np.random.uniform() * 2 * np.pi
            rotate = Rotation.from_euler(seq='z', angles=theta, degrees=False)
            rn = rotate.apply(ro)
            atoms.set_positions(rn)
        rn = atoms.get_positions()
        world.broadcast(rn, 0)
        atoms.set_positions(rn)
        return atoms.get_positions()

    def log(self, step, En, Emin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        if En == float('inf'):
            self.logfile.write('%s: step %d, optimization FAIL to converge or CAN NOT maintain one cluster\n'
                               % (name, step))
            self.logfile.flush()
        else:
            self.logfile.write('%s: step %d, energy %15.6f, emin %15.6f\n'
                               % (name, step, En, Emin))
            self.logfile.flush()
