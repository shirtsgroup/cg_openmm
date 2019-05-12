class Model(object):
    # Build a model object

    def __init__(self,positions):
        self.positions = positions
        self.chains = []
        self._current_chain = None
        self.chains_by_id = {}
        self.connects = []

        add_new_elements()

        chain = Chain()
        particle_index = 0

        for monomer_index in range(polymer_length):

          monomer = Monomer(name=str("M"+str(monomer_index)),number=monomer_index)

          for backbone_bead in range(backbone_length):

            particle = Particle(particle_name='CG1', particle_number=particle_index, monomer_name=str("M"+str(monomer_index)), monomer_number=monomer_index, position=positions[particle_index], element_symbol='CG1',chain_id=chain.chain_id)
            monomer._add_particle(particle)
            particle_index = particle_index + 1
            
            if backbone_bead in sidechain_positions:
 
              particle = Particle(particle_name='CG2', particle_number=particle_index, monomer_name=str("M"+str(monomer_index)), monomer_number=monomer_index, position=positions[particle_index], element_symbol='CG2',chain_id=chain.chain_id)
              monomer._add_particle(particle)
              particle_index = particle_index + 1
          chain._add_monomer(monomer)
        self._add_chain(chain)

    def _add_chain(self, chain):
        self.chains.append(chain)
        self._current_chain = chain
        if not chain.chain_id in self.chains_by_id:
            self.chains_by_id[chain.chain_id] = chain

    def get_chain(self, chain_id):
        return self.chains_by_id[chain_id]

    def chain_ids(self):
        return self.chains_by_id.keys()

    def __contains__(self, chain_id):
        return self.chains_by_id.__contains__(chain_id)

    def __getitem__(self, chain_id):
        return self.chains_by_id[chain_id]

    def __iter__(self):
        return iter(self.chains)

    def iter_chains(self):
        for chain in self:
            yield chain

    def iter_monomers(self):
        for chain in self:
            for res in chain.iter_monomers():
                yield res

    def iter_particles(self):
        for chain in self:
            for particle in chain.iter_particles():
                yield particle

    def iter_positions(self):
        for chain in self:
            for loc in chain.iter_positions():
                yield loc

class Chain(object):
    def __init__(self, chain_id="A"):
        self.chain_id = chain_id
        self.monomers = []
        self.monomers_by_number = {}

    def _add_monomer(self, monomer):
        if len(self.monomers) == 0:
            monomer.is_first_in_chain = True
        self.monomers.append(monomer)
        self._current_monomer = monomer
        # only store the first monomer with a particular key
        if monomer.number not in self.monomers_by_number:
            self.monomers_by_number[monomer.number] = monomer

    def get_monomer(self, monomer_number, insertion_code=' '):
        return self.monomers_by_num_icode[str(monomer_number) + insertion_code]

    def __contains__(self, monomer_number):
        return self.monomers_by_number.__contains__(monomer_number)

    def __getitem__(self, monomer_number):
        """Returns the FIRST monomer in this chain with a particular monomer number"""
        return self.monomers_by_number[monomer_number]

    def __iter__(self):
        for res in self.monomers:
            yield res

    def iter_monomers(self):
        for res in self:
            yield res

    def iter_particles(self):
        for res in self:
            for particle in res:
                yield particle;

    def iter_positions(self):
        for res in self:
            for loc in res.iter_positions():
                yield loc

    def __len__(self):
        return len(self.monomers)

class Monomer(object):

    def __init__(self, name, number):
        self.locations = {}
        self._name = name
        self.number = number
        self.particles = []
        self.particles_by_name = {}

    def _add_particle(self, particle):
        self.locations = particle.locations
        self.particles_by_name[particle.name] = particle
        self.particles.append(particle)

    def get_name(self):
        return self._name
    name = property(get_name, doc='monomer name')

    def get_particle(self, particle_name):
        return self.particles_by_name[particle_name]

    def __iter__(self):
        for particle in self.iter_particles():
            yield particle

    def iter_particles(self):
        for particle in self.particles:
           yield particle

    def iter_positions(self):
        """
        Returns one position per particle, even if an individual particle has multiple positions.

        """
        for particle in self:
                yield particle.position

    def __len__(self):
        return len(self.particles)

class Particle(object):
    """Particle represents one particle in our model.
    """

    def __init__(self, particle_name, particle_number, monomer_name, monomer_number, position, element_symbol, chain_id):
        """Create a new particle
        """
        self.name = particle_name
        self.number = particle_number
        self.monomer_name = monomer_name
        self.monomer_number = monomer_number
        self.chain_id = chain_id

        self.locations = position
        self.element = element_symbol

    def iter_locations(self):
        """
        Iterate over Particle.Location objects for this particle, including primary location.

        """
        for alt_loc in self.locations:
            yield self.locations[alt_loc]

    def iter_positions(self):
        """
        Iterate over particle positions.  Returns Quantity(Vec3(), unit) objects, unlike
        iter_locations, which returns Particle.Location objects.
        """
        for loc in self.iter_locations():
            yield loc.position

    def iter_coordinates(self):
        """
        Iterate over x, y, z values of primary particle position.

        """
        for coord in self.position:
            yield coord

    def get_location(self, location_id=None):
        id = location_id
        if (id == None):
            id = self.default_location_id
        return self.locations[id]

    def set_location(self, new_location, location_id=None):
        id = location_id
        if (id == None):
            id = self.default_location_id
        self.locations[id] = new_location
    location = property(get_location, set_location, doc='default Particle.Location object')

    def get_position(self):
        return self.location.position

    def set_position(self, coords):
        self.location.position = coords

    def get_x(self): return self.position[0]
    x = property(get_x)

    def get_y(self): return self.position[1]
    y = property(get_y)

    def get_z(self): return self.position[2]
    z = property(get_z)

    class Location(object):
        """
        Inner class of Particle for holding alternate locations
        """
        def __init__(self, position, monomer_name):
            self.position = position

        def __iter__(self):
            for coord in self.position:
                yield coord

        def __str__(self):
            return str(self.position)


def build_topology(positions):
        # Create topology.
        topology = Topology()

        # Build the model
        model = Model(positions)

        # Build the topology

        particleByNumber = {}
        for chain in model.iter_chains():
            c = topology.addChain(chain.chain_id)
            for monomer in chain.iter_monomers():
                resName = monomer.get_name()
                for particle in monomer.iter_particles():
                    particleName = particle.name
                    particleName = particleName.strip()
                    element = particle.element
                    newParticle = topology.addAtom(particleName, element, monomer, str(particle.number))
        return(topology)

