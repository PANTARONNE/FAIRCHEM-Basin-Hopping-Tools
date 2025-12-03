import warnings
import argparse
import yaml
from ase.io import write
from bh_tools.basin_hopping_simple import BasinHoppingSimple
from create_tools.cleave_vacuum import CleaveCreator
from create_tools.inverse_chain import InverseChainBuilder
from constrain_tools.constrain_slab import ConstrainLayers

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    args = parse_args()
    conf = load_config(args.config)

    cc = CleaveCreator(atom_symbol=conf["metal"],
                       direction=(conf["lattice_plane"]["a"], conf["lattice_plane"]["b"], conf["lattice_plane"]["c"]),
                       xyz=(conf["expand"]["x"], conf["expand"]["y"], conf["expand"]["z"]),
                       vacuum=conf["vacuum"],
                       potential_path=conf["potential_path"],
                       job_type=conf["job_type_opt"])
    Slab = cc.build_surface()
    icb = InverseChainBuilder(cluster_comp=conf["oxide"], slab=Slab, m_num=conf["m_num"], o_num=conf["o_num"],
                              oxide_cif=f"%s%s" % (conf["cif_path"], conf["oxide"] + '.cif'), dist=conf["dist"])
    Model = icb.build_inverse()
    write(f"%s%s-%s-m%do%d-orig.cif" % (conf["out_path"], conf["metal"], conf["oxide"], conf["m_num"], conf["o_num"]),
          Model, format='cif')
    cl = ConstrainLayers(Model, sub_layers=conf["bh_constrain_layers"])
    Model = cl.set_constrain()
    bh_sa = BasinHoppingSimple(system=Model, temp=conf["temp"], hop_steps=conf["hop_steps"],
                               disturbance=conf["disturbance"], f_max=conf["f_max"],
                               step_max=conf["step_max"], log_file_path=f"%schain-bh.log" % conf["out_path"],
                               potential_path=conf["potential_path"], job_type=conf["job_type_bh"])
    Opted = bh_sa.run_basin_hopping_sa()
    write(f"%s%s-%s-m%do%d-opt.cif" % (conf["out_path"], conf["metal"], conf["oxide"], conf["m_num"], conf["o_num"]),
          Opted, format='cif')
