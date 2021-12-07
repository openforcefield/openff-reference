import openmm
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import (ForceField,
                                                    ImproperTorsionHandler,
                                                    ProperTorsionHandler)

from patches import _split_torsion_force


def test_split_torsion_force():
    molecule = Molecule.from_smiles("C1=CC=CC=C1")
    topology = molecule.to_topology()
    sage = ForceField("openff-2.0.0.offxml")
    system = sage.create_openmm_system(topology)

    original_torsion_force = [
        f for f in system.getForces() if type(f) == openmm.PeriodicTorsionForce
    ][0]
    new_proper_force, new_improper_force = _split_torsion_force(
        original_torsion_force,
        sage["ProperTorsions"],
        sage["ImproperTorsions"],
        topology,
    )

    assert (
        new_proper_force.getNumTorsions() + new_improper_force.getNumTorsions()
        == original_torsion_force.getNumTorsions()
    )
