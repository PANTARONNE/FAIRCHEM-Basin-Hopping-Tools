from ase import Atoms, units
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.io import Trajectory


class SimulatedAnnealing:
    def __init__(self, system: Atoms, t_heat: float, t_cool: float,
                 heat_steps: int, cool_steps: int, stable_steps: int,
                 dt_fs: float, friction: float, t_damp: float, t_chain: int, t_loop: int, log_interval: int,
                 traj_file_path: str, potential_path: str, job_type: str):
        """
        Run simulated annealing MD based on Nose-Hoover NVT system.
            1. Heat to T_heat;
            2. Cool down to T_cool linearly;
            3. Run a short MD at T_cool to ensure stability.
        :param system: ase.Atoms object expected
        :param t_heat: heat temperature
        :param t_cool: cool temperature
        :param heat_steps: heating steps
        :param cool_steps: cooling steps
        :param dt_fs: length of single step
        :param t_damp: damping timescale
        :param traj_file_path: traj file path
        :param potential_path: potential file path
        :param job_type: uma job type
        """
        self.atoms = system
        self.temp_max = t_heat
        self.temp_min = t_cool
        self.heat_n = heat_steps
        self.cool_n = cool_steps
        self.stable_n = stable_steps
        self.dt = dt_fs
        self.friction = friction
        self.damp = t_damp
        self.chain = t_chain
        self.loop = t_loop
        self.interval = log_interval
        self.out_path = traj_file_path
        self.potential = potential_path
        self.job = job_type
        self.model = load_predict_unit(path=self.potential, device="cuda")
        self.calculator = FAIRChemCalculator(self.model, task_name=self.job)
        self.atoms.calc = self.calculator
        self.traj = None
        self.vessel = None

    def run_heat(self):
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temp_max)  # init speed

        self.traj = Trajectory(self.out_path, 'w', self.atoms)
        self.vessel = Langevin(atoms=self.atoms, timestep=self.dt * units.fs,
                               temperature_K=self.temp_max, friction=self.friction)
        self.vessel.attach(self.traj.write, interval=1)
        self.vessel.attach(self.print_vessel_status, interval=self.interval)
        print("========== Heating Start ============")
        self.vessel.run(steps=self.heat_n)

    def run_cool(self):
        print("========== Cooling Start ============")
        for i in range(self.cool_n):
            frac = i / float(self.cool_n)
            t_now = self.temp_max + frac * (self.temp_min - self.temp_max)

            # MaxwellBoltzmannDistribution(self.atoms, temperature_K=t_now)
            self.vessel = NoseHooverChainNVT(atoms=self.atoms, timestep=self.dt * units.fs, temperature_K=t_now,
                                             tdamp=self.damp, tchain=self.chain, tloop=self.loop)
            self.vessel.attach(self.conditional_writer, interval=1)
            if (i + 1) % self.interval == 0:
                self.print_cooler_status(step=(i+1))
            self.vessel.run(steps=1)

        self.vessel = NoseHooverChainNVT(atoms=self.atoms, timestep=self.dt * units.fs, temperature_K=self.temp_min,
                                         tdamp=self.damp, tchain=self.chain, tloop=self.loop)
        self.vessel.attach(self.traj.write, interval=1)
        self.vessel.attach(self.print_vessel_status, interval=self.interval)
        print("============ Finalizing =============")
        self.vessel.run(steps=self.stable_n)

    def run_sa(self):
        self.run_heat()
        self.run_cool()
        return self.atoms

    def print_vessel_status(self):
        e_pot = self.atoms.get_potential_energy()
        e_kin = self.atoms.get_kinetic_energy()
        t = e_kin / (1.5 * (len(self.atoms) - len(self.atoms.constraints[0].get_indices())) * units.kB)
        step = self.vessel.get_number_of_steps()
        print(f"Step {step: 6d}  E_pot = {e_pot: .6f} eV  E_kin = {e_kin: .6f} eV  T = {t: .1f} K")

    def print_cooler_status(self, step):
        e_pot = self.atoms.get_potential_energy()
        e_kin = self.atoms.get_kinetic_energy()
        t = e_kin / (1.5 * (len(self.atoms) - len(self.atoms.constraints[0].get_indices())) * units.kB)
        print(f"Step {step: 6d}  E_pot = {e_pot: .6f} eV  E_kin = {e_kin: .6f} eV  T = {t: .1f} K")

    def conditional_writer(self):
        step = self.vessel.get_number_of_steps()
        if step > 0:
            self.traj.write(self.atoms)
