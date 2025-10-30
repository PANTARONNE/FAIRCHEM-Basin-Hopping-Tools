from typing import IO, Type, Union
import numpy as np
from ase import Atoms, units
from ase.optimize.fire import FIRE
from ase.optimize.optimize import Dynamics, Optimizer
from ase.optimize.basin import BasinHopping


class BasinHoppingEarlyStop(BasinHopping):
    def __init__(
        self,
        atoms: Atoms,
        temperature: float = 100 * units.kB,
        optimizer: Type[Optimizer] = FIRE,
        fmax: float = 0.1,
        step_max: int = 200,
        dr: float = 0.1,
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
        self.kT = temperature
        self.optimizer = optimizer
        self.fmax = fmax
        self.step_max = step_max
        self.dr = dr
        if adjust_cm:
            self.cm = atoms.get_center_of_mass()
        else:
            self.cm = None

        self.optimizer_logfile = optimizer_logfile

        Dynamics.__init__(self, atoms, logfile)
        self.initialize()

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
            else:
                self.energy = self.optimizable.get_value()

        return self.energy

    def log(self, step, En, Emin):
        if self.logfile is None:
            return
        name = self.__class__.__name__
        if En == float('inf'):
            self.logfile.write('%s: step %d, structural optimization FAIL to converge\n'
                               % (name, step))
            self.logfile.flush()
        else:
            self.logfile.write('%s: step %d, energy %15.6f, emin %15.6f\n'
                               % (name, step, En, Emin))
            self.logfile.flush()
