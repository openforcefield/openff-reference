from copy import deepcopy
from typing import Tuple

import openmm
from openff.toolkit.topology.topology import Topology
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ImproperTorsionHandler, ProperTorsionHandler)


def _split_torsion_force(
    combined_torsion_force: openmm.openmm.PeriodicTorsionForce,
    proper_torsion_handler: ImproperTorsionHandler,
    improper_torsion_handler: ImproperTorsionHandler,
    topology: Topology,
) -> Tuple[openmm.openmm.PeriodicTorsionForce, openmm.openmm.PeriodicTorsionForce]:
    """
    Split the combined torsion force into two separate forces, one for proper
    torsions and one for improper torsions.
    """
    new_proper_torsion_force = openmm.PeriodicTorsionForce()
    new_improper_torsion_force = openmm.PeriodicTorsionForce()

    proper_quartets = [
        value.environment_match.topology_atom_indices
        for value in proper_torsion_handler.find_matches(topology).values()
    ]

    # A more thorough solution would compare the matches found for improper torsions,
    # but it's somewhat complicated by turning each into a trefoil, so mapping backwards
    # from the OpenMM force is not so straightforward. This solution assumes that
    # all atom quartets found in the force but not found in the proper torsion matches
    # are by deduction impropers. This is a weak point in this implementation.
    improper_quartets = [
        value.environment_match.topology_atom_indices
        for value in improper_torsion_handler.find_matches(topology).values()
    ]

    for i in range(combined_torsion_force.getNumTorsions()):
        atom_indices = tuple(combined_torsion_force.getTorsionParameters(i)[:4])
        if (
            atom_indices in proper_quartets
            or tuple(reversed(atom_indices)) in proper_quartets
        ):
            new_proper_torsion_force.addTorsion(
                *combined_torsion_force.getTorsionParameters(i)
            )
        else:
            # if (
            #     atom_indices in improper_quartets
            #     or tuple(reversed(atom_indices)) in improper_quartets
            # ):
            new_improper_torsion_force.addTorsion(
                *combined_torsion_force.getTorsionParameters(i)
            )

    return new_proper_torsion_force, new_improper_torsion_force


def _remove_vdw_cutoff(
    openmm_system: openmm.System, coulomb_14: int = 0.8333333333, vdw_14: int = 0.5
) -> openmm.System:
    """
    Handle the case in which the toolkit sets all non-bonded interactions to
    NonbondedForce.NoCutoff when the methods are defined in the force field as
    "cutoff" and "PME" respectively. Split these interactions out into a new
    NonbondedForce for the electrostatics component, a CustomNonbondedForce for
    the vdW interactions, and CustomBondForce for the 1-4 interactions of the
    vdW component. The 1-4 interactions of the electrostatics components are
    wrapped up into the NonbondedForce since there is no need to split them out.
    """
    for i in range(openmm_system.getNumForces()):
        force = openmm_system.getForce(i)
        if type(force) == openmm.NonbondedForce:
            original_nonbonded_force = force
            original_nonbonded_force_copy = deepcopy(force)
            original_nonbonded_force_index = i

    #  0 == openmm.NonbondedForce.NoCutoff
    assert original_nonbonded_force.getNonbondedMethod() == 0

    bonds = [
        original_nonbonded_force.getExceptionParameters(i)[:2]
        for i in range(original_nonbonded_force.getNumExceptions())
    ]

    new_nonbonded_force = openmm.NonbondedForce()

    lj_12_6_expression = "4*epsilon*((sigma/r)^12-(sigma/r)^6)"
    lorentz_berthelot_expression = (
        "sigma=(sigma1+sigma2)/2; epsilon=sqrt(epsilon1*epsilon2);"
    )

    vdw_force = openmm.CustomNonbondedForce(
        lj_12_6_expression + ";" + lorentz_berthelot_expression
    )
    vdw_force.addPerParticleParameter("sigma")
    vdw_force.addPerParticleParameter("epsilon")

    vdw_14_force = openmm.CustomBondForce(lj_12_6_expression)
    vdw_14_force.addPerBondParameter("sigma")
    vdw_14_force.addPerBondParameter("epsilon")
    vdw_14_force.setUsesPeriodicBoundaryConditions(True)

    new_nonbonded_force.setName("Electrostatics")
    vdw_force.setName("VdW")
    vdw_14_force.setName("VdW 1-4")
    openmm_system.addForce(new_nonbonded_force)
    openmm_system.addForce(vdw_force)
    openmm_system.addForce(vdw_14_force)

    for i in range(original_nonbonded_force_copy.getNumParticles()):
        charge, sigma, epsilon = original_nonbonded_force_copy.getParticleParameters(i)

        new_nonbonded_force.addParticle(charge, 0.0, 0.0)
        vdw_force.addParticle([sigma, epsilon])

    new_nonbonded_force.createExceptionsFromBonds(
        bonds=bonds,
        coulomb14Scale=coulomb_14,
        lj14Scale=0.0,
    )

    for i in range(new_nonbonded_force.getNumExceptions()):
        (p1, p2, q, sigma, epsilon) = new_nonbonded_force.getExceptionParameters(i)

        if q._value == 0:
            pass
        else:
            sigma1, epsilon1 = vdw_force.getParticleParameters(p1)
            sigma2, epsilon2 = vdw_force.getParticleParameters(p2)

            sigma_14 = (sigma1 + sigma2) * 0.5
            epsilon_14 = (epsilon1 * epsilon2) ** 0.5 * vdw_14

            vdw_14_force.addBond(p1, p2, [sigma_14, epsilon_14])

        vdw_force.addExclusion(p1, p2)

    openmm_system.removeForce(original_nonbonded_force_index)

    return openmm_system
