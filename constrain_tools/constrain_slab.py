from ase.constraints import FixAtoms


class ConstrainLayers:
    def __init__(self, model, sub_layers):
        self.struc = model
        self.layers = sub_layers

    def set_constrain(self):
        z_pos = self.struc.get_positions()[:, 2]
        z_sorted = sorted(set(round(z, 3) for z in z_pos))
        bottom_layers = z_sorted[:self.layers]
        mask = [round(z, 3) in bottom_layers for z in z_pos]
        constraint = FixAtoms(mask)
        self.struc.set_constraint(constraint)
        print("========== Constraint Set ===========")
        return self.struc
