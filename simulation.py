'''
Class implementing the active particle molecular dynamics simulation with
the Velocity-Verlet-Scheme

Mishael Derla, FAU Erlangen-Nürnberg, winter term 2024/2025
'''

from time import perf_counter

import numpy

from xyz_file_io import xyz_dump

from physics_settings import PhysicsSettings
from physics import repulsion
from initial_conditions import overlap_free_configuration
from initial_conditions import sum_zero_unit_vectors
from initial_conditions import two_directions_sum_zero_unit_vectors


class Simulation:
    '''
    Verlet-Integrator for the equations of motion
    '''

    CONTINUOUS_PROPULSION_ANGLES = 'continuous_propulsion_angles'
    ONLY_TWO_OPPOSITE_PROPULSIONS = 'only_two_opposite_propulsions'

    @staticmethod
    def run(
        physics_settings: PhysicsSettings,
        step_count: int,
        xyz_dump_filepath: str,
        only_two_propulsions:bool=False
        ):
        '''
        Run simulation with the passed settings
        '''

        # information string used to not loose simulation information
        information_string = physics_settings.information_string()

        # system dimensions
        n = physics_settings.particle_number
        d = physics_settings.spatial_dimensions
        l = physics_settings.box_sidelength

        # time step
        dt = physics_settings.time_step

        # packing fraction and particle diameter
        sigma = physics_settings.particle_diameter

        # particle mass
        m = physics_settings.particle_mass

        # friction constant
        gamma = physics_settings.friction_constant

        # forces
        propulsion_force_magnitude = physics_settings.propulsion_force
        pair_force_magnitude = physics_settings.pair_force

        print('Searching for overlap free configuration...')
        # initializing overlap free (if possible) in box [0,1]^d, with zero initial velocity
        positions = overlap_free_configuration(
            n, d,
            l,
            sigma,
            0.005 * sigma,
            max_iterations=1000
        )

        # vanishing initial velociry
        velocities = numpy.zeros(positions.shape)

        # initializing propulsions of equal length that sum to zero
        print(f'Searching for {n} unit vectors summing to zero...')
        unit_vectors = two_directions_sum_zero_unit_vectors(n, d) if only_two_propulsions else sum_zero_unit_vectors(n, d)
        self_propulsion = propulsion_force_magnitude * unit_vectors
        # self_propulsion = propulsion_force_magnitude * unit_vectors

        # first computation of the forces
        forces = repulsion(positions, l, sigma, pair_force_magnitude) + self_propulsion

        print('Commencing molecular dynamics simulation...')
        perf_counter_start = perf_counter()

        for step in range(step_count):

            if step % int(step_count/10) == 0 or step + 1 == step_count:
                progress = int(100*step/step_count)
                elapsed_time = perf_counter() - perf_counter_start
                print(f'Progress {progress:3}% ({elapsed_time/60:.1f} min)')

            # computing the next positions with the Verlet scheme, while ...
            # ... enforcing periodic boundary conditions
            positions_next = (positions + velocities * dt + (1/2)*(forces/m)*dt**2) % l

            # computing all the forces with the next positions, where ...
            # ... repulsive forces are kept separate in order to
            repulsion_next = repulsion(positions_next, l, sigma, pair_force_magnitude)

            # NOTE: friction may be badly implemented
            forces_next = repulsion_next + self_propulsion - gamma * velocities

            # computing the next velocities
            velocities_next = velocities + ((forces + forces_next)/2)*dt

            # dumping the current positions, where first labels are ...
            # ... chosen for each particle based on whether they are ...
            # ... involved in a repulsion ...
            particle_labels = [
                'colliding' if colliding else 'free'
                for colliding in numpy.linalg.norm(repulsion_next, axis=1) > 0.01
            ]

            # ... the the actual xyz-dump is done
            xyz_dump(
                xyz_dump_filepath,
                particle_labels,
                positions,
                propulsions=self_propulsion,
                radius=sigma/2,
                comment=f'step {step}; {information_string}'
            )

            # making the next last postisitions the current, and ...
            # ... the current positions the next, and enforcing ...
            # ... periodic boundary condtions
            positions = positions_next
            velocities = velocities_next
            forces = forces_next

        perf_counter_end = perf_counter()
        print(f'Done. ({(perf_counter_end - perf_counter_start)/60:.1f} min)')


if __name__ == '__main__':

    import os

    # packing fraction and particle number at which we ...
    # ... study alignment
    PHI = 0.1
    M = 0.5 # mass
    N = 50

    # target directory ...
    target_directory = f'alignment_distribution_function/m={M}_phi={PHI}'
    os.makedirs(target_directory, exist_ok=True)

    for sample in range(10):

        filename = f'm={M}_phi={PHI}_alignment_{sample}_N={N}.xyz'

        Simulation.run(
            physics_settings=PhysicsSettings.standard(
                particle_number=N,
                spatial_dimensions=2,
                phi=PHI,
                particle_mass=M
            ),
            step_count=5000,
            xyz_dump_filepath=f'{target_directory}/{filename}'
        )
