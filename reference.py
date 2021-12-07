from copy import deepcopy
from typing import Dict

import openmm
import pandas
from openff.interchange.drivers.openmm import get_openmm_energies
from openff.toolkit.tests.utils import get_context_potential_energy
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit as openmm_unit

from patches import _remove_vdw_cutoff, _split_torsion_force


def _canonicalize_openmm_force_name(force_name: str) -> str:
    mapping = {
        "PeriodicTorsionForce": "E_torsion",
        "NonbondedForce": "E_nonbonded",
        "HarmonicBondForce": "E_bond",
        "HarmonicAngleForce": "E_angle",
    }
    return mapping[force_name]


def _generate_reference(reference_inputs: Dict) -> pandas.DataFrame:

    molecule = Molecule.from_file(reference_inputs["ligand_file"])
    positions = molecule.conformers[0]
    topology = molecule.to_topology()

    force_field = ForceField(reference_inputs["force_field"])
    openmm_system = force_field.create_openmm_system(topology)

    for i, force in enumerate(openmm_system.getForces()):
        if type(force) == openmm.PeriodicTorsionForce:
            original_torsion_force = force
            break

    new_proper_force, new_improper_force = _split_torsion_force(
        original_torsion_force,
        force_field["ProperTorsions"],
        force_field["ImproperTorsions"],
        topology,
    )

    new_proper_force.setName("Proper Torsions")
    new_improper_force.setName("Imroper Torsions")

    openmm_system.removeForce(i)
    openmm_system.addForce(new_proper_force)
    openmm_system.addForce(new_improper_force)

    for force in openmm_system.getForces():
        if type(force) == openmm.HarmonicAngleForce:
            force.setName("Angles")
        if type(force) == openmm.HarmonicBondForce:
            force.setName("Bonds")

    force_groups: Dict[int, str] = dict()

    for i, force in enumerate(openmm_system.getForces()):
        force_groups[i] = force.getName()
        force.setForceGroup(i)

    platform = openmm.Platform.getPlatformByName("Reference")
    integrator = openmm.VerletIntegrator(1.0 * openmm_unit.femtoseconds)

    try:
        off_context = openmm.Context(openmm_system, deepcopy(integrator), platform)
    except:
        openmm_system.removeForce(openmm_system.getNumForces() - 2)

    off_context = openmm.Context(openmm_system, deepcopy(integrator), platform)
    toolkit_energy = get_context_potential_energy(
        off_context, positions, by_force_group=True
    )

    return pandas.DataFrame.from_dict(
        dict(
            (
                force_groups[k],
                [round(val.value_in_unit(openmm_unit.kilojoule_per_mole), 6)],
            )
            for k, val in toolkit_energy.items()
        )
    )


reference_inputs = [
    {
        "force_field": "openff_unconstrained-2.0.0.offxml",
        "ligand_file": "data/ethanol.sdf",
        "periodic": False,
        "box_vectors": None,
    },
    {
        "force_field": "openff_unconstrained-2.0.0.offxml",
        "ligand_file": "data/lig_15.sdf",
        "periodic": False,
        "box_vectors": None,
    },
]

results = pandas.DataFrame()
for input in reference_inputs:
    results = pandas.concat([results, _generate_reference(input)])

results.insert(0, "Total Energy", results.sum(axis=1))

print(results)
