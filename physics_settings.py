'''
Class containing bookkeeping classes for the active particle simulations

Mishael Derla, FAU Erlangen-Nürnberg, winter term 2024/2025
'''


from dataclasses import dataclass

import numpy

# for gamma function
from scipy.special import gamma as euler_gamma_function



@dataclass
class PhysicsSettings:
    '''
    Class containing information about the simulation's physical dimensions
    and units
    '''

    # dimensions
    particle_number: int
    spatial_dimensions: int

    # packing fraction
    packing_fraction: float

    # units
    #   system size and timestep
    box_sidelength: float
    time_step: float
    #   driving forces and particle characterization
    propulsion_force: float
    pair_force: float
    particle_mass: float
    #   friction
    terminal_speed: float


    @staticmethod
    def standard(
        particle_number: int,
        spatial_dimensions: int,
        phi: float,
        f:float=10,
        v_inf_to_v_br:float=1,
        particle_mass:float=1
    ) -> 'PhysicsSettings':
        '''
        Returns a PhysicsSettings instance whose parameters are set
        in the manner we have grown accustomed to

        Args:
            particle_number:
                the number of active particles
            spatial_dimensions:
                the number of spatial dimensions of the domain the
                active partiles move in
            phi:
                packing fraction of the active particles
            f:
                the ratio of pair to self propulsion force
            v_inf_to_v_br:
                the ratio of terminal to breaking velocity
        '''

        d = spatial_dimensions

        # sidelength and system volume
        box_sidelength = 1
        volume = box_sidelength**d

        # packing fraction and corresponding particle diameter
        #   (a) unit ball volume
        unit_ball_volume = numpy.pi**(d/2) / euler_gamma_function(1+d/2)
        #   (b) Because phi = B * (sigma/2)**d * N / V, rearranging results in
        #       sigma = 2 * (phi * V / (N * B))**(1/d)
        sigma = 2 * (phi * volume / (particle_number * unit_ball_volume)) ** (1/d)

        # time step
        dt = 0.001

        # particle mass relative to the unit mass m_0
        m = particle_mass

        # NOTE: when varying mass, one needs to keep everything else ...
        # ... fixed, so m is only given in terms of unit mass
        m_0 = 1

        # self-propulsion constant. NOTE: while varying mass, F_self and ...
        # ... F_pair should stay fixed, thus we use m_0 instead of m
        propulsion_force = 0.0001 * m_0*sigma/(dt**2) # 0.0001 * m*L/(dt**2)

        # repulsion constant and corresponding breaking velocity
        pair_force = f * propulsion_force
        # NOTE: while varying mass, the fiction constant should stay fixed ...
        # ... thus we use m_0 instead of m
        v_break = (1/2) * numpy.sqrt(2 * pair_force * sigma / m_0)

        # terminal velocity and implied friction constant & friction relaxation time
        v_infty = v_inf_to_v_br * v_break # v_break # sigma / tau

        return PhysicsSettings(
            particle_number,
            spatial_dimensions,
            phi,
            box_sidelength,
            dt,
            propulsion_force,
            pair_force,
            m,
            v_infty
        )


    @property
    def volume(self):
        '''
        Returns the system volume implied by the box sidelength and the number of
        spatial dimensions
        '''
        return self.box_sidelength**self.spatial_dimensions

    @property
    def particle_diameter(self):
        '''
        Computes the particle diameter implied by the packing fraction, the system
        volume, the number of particles and the number of spatial dimensions
        '''
        d = self.spatial_dimensions
        phi = self.packing_fraction
        volume = self.volume
        particle_number = self.particle_number
        # packing fraction and corresponding particle diameter
        #   (a) unit ball volume
        unit_ball_volume = numpy.pi**(d/2) / euler_gamma_function(1+d/2)
        #   (b) Because phi = B * (sigma/2)**d * N / V, rearranging results in
        #       sigma = 2 * (phi * V / (N * B))**(1/d)
        return 2 * (phi * volume / (particle_number * unit_ball_volume)) ** (1/d)


    @property
    def breaking_speed(self):
        '''
        Computes the lab frame speed necessary to enable breaking the pair potential
        barrier in particle-particle collisions (to the very least via head on collisions)
        '''
        return (1/2) * numpy.sqrt(2 * self.pair_force * self.particle_diameter / self.particle_mass)

    @property
    def mass_timescale(self):
        '''
        Combines self-propulsion, particle diameter and particle mass to a timescale
        '''
        return numpy.sqrt(self.particle_mass * self.particle_diameter / self.propulsion_force)


    @property
    def friction_constant(self):
        '''
        Computes the friction constant implied by self-propulsion force and terminal
        speed
        '''
        return self.propulsion_force / self.terminal_speed

    @property
    def friction_timescale(self):
        '''
        Computes the characteristic time with which a particle velocity relaxes to terminal velocity
        '''
        return self.particle_mass / self.friction_constant
    

    def information_string(self):
        '''
        Compiles a string with all information in the PhysicsSettings instance
        relevant to the analysis of the simulation data of in the format
        "property_1=value_1, property_2=value_2, ..."
        '''

        # joining all properties and their value in a comma separated string of ...
        # ... the format 'property_1=value_1, property_2=value_2, ...'
        return ', '.join(
            f'{attribute}={getattr(self, attribute)}'
            for attribute in dir(self)
            if not (

                # excluding the __...__ technical attributes
                attribute.startswith('__')
                or attribute.endswith('__')

                # excluding the PhysicsSettings.standard() static method and the
                # ... present PhysicsSettings.information_string() method
                or attribute in ['standard', 'information_string']
            )
        )


if __name__ == '__main__':

    # NOTE: testing
    test = PhysicsSettings.standard(10, 2, 0.2, f=60)
    print(test.information_string())
