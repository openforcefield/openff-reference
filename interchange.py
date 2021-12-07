from pprint import pprint

from openff.interchange.components.interchange import Interchange
from openff.interchange.drivers import get_openmm_energies
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit as openmm_unit

molecule = Molecule.from_file("data/lig_15.sdf")

sage = ForceField("openff-2.0.0.offxml")

topology = molecule.to_topology()

topology.box_vectors = [8, 8, 8] * openmm_unit.nanometer

out = Interchange.from_smirnoff(sage, topology)
out.positions = molecule.conformers[0]

pprint(get_openmm_energies(out).energies)
