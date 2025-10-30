import warnings
import argparse
import yaml
from sa_tools.simulated_annealing import SimulatedAnnealing
from create_tools.cleave_vacuum import CleaveCreator
from create_tools.inverse_sphere import InverseSphereBuilder
from constrain_tools.constrain_slab import ConstrainLayers

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def oxide2cif(oxide_name: str):
    dic = {
        "ZrO2": "./oxide_models/ZrO2_mp-2858_primitive.cif",
        "Al2O3": "./oxide_models/Al2O3_mp-1143_primitive.cif",
        "CeO2": "./oxide_models/CeO2_mp-20194_primitive.cif",
        "In2O3": "./oxide_models/In2O3_mp-22598_primitive.cif",
        "La2O3": "./oxide_models/La2O3_mp-2292_primitive.cif",
        "TiO2": "./oxide_models/TiO2_mp-554278_primitive.cif",
        "Y2O3": "./oxide_models/Y2O3_mp-2652_primitive.cif",
    }
    return dic[oxide_name]


if __name__ == '__main__':
    from ase.visualize import view
    args = parse_args()
    conf = load_config(args.config)

    cc = CleaveCreator(atom_symbol=conf["metal"],
                       direction=(conf["lattice_plane"]["a"], conf["lattice_plane"]["b"], conf["lattice_plane"]["c"]),
                       xyz=(conf["expand"]["x"], conf["expand"]["y"], conf["expand"]["z"]),
                       vacuum=conf["vacuum"],
                       potential_path=conf["potential_path"],
                       job_type=conf["job_type_opt"])
    Slab = cc.build_surface()
    isb = InverseSphereBuilder(slab=Slab, oxide_cif=oxide2cif(conf["oxide"]), rcut=conf["rcut"], dist=conf["dist"])
    Model = isb.build_inverse()
    cl = ConstrainLayers(Model, sub_layers=conf["md_constrain_layers"])
    Model = cl.set_constrain()
    Sa = SimulatedAnnealing(Model, t_heat=conf["t_heat"], t_cool=conf["t_cool"], heat_steps=conf["heat_steps"],
                            cool_steps=conf["cool_steps"], stable_steps=conf["stable_steps"], dt_fs=conf["dt_fs"],
                            friction=conf["friction"], t_damp=conf["t_damp"],
                            t_chain=conf["t_chain"], t_loop=conf["t_loop"],
                            log_interval=conf["log_interval"], traj_file_path=conf["traj_path"],
                            potential_path=conf["potential_path"], job_type=conf["job_type_md"])
    Model = Sa.run_sa()
