'''
Functions for preparing the initial conditions of the system

Mishael Derla, FAU Erlangen-Nürnberg, winter term 2024/2025
'''

import warnings
import numpy

from physics import pbc_iterator


from xyz_file_io import xyz_dump


def overlap_free_configuration(
        particle_number: int, spatial_dimensions: int,
        box_sidelength:float,
        particle_diameter:float,
        drift_speed:float,
        max_iterations:int=1000,
        xyz_dump_filepath:str=None
    ) -> numpy.ndarray:
    '''
    Creates an initial configuration with significantly reduced particle overlaps

    Args:
        particle_number:
            the number of particles
        spatial_dimensions:
            the number of spatial dimensions of the periodic boundary
            box that the particles move in
        box_sidelength:
            sidelength of the periodic boundary box that the particles
            move in
        max_iteration:
            the maximum numver of iterations that the configuration searching
            dynamics will go through. The search might terminate well before
            that, if (most likely) an absorbing state (no overlaps) was found
        drift_speed:
            the step size of the configuration space search
            
    Returns:
        an numpy.ndarray of positions of a configuratioln with significantly reduced overlaps
    '''

    # random positions
    positions = numpy.random.rand(particle_number, spatial_dimensions)

    for iteration in range(max_iterations):

        # TODO: generalize progress bar, better user feedback
        if iteration % int(max_iterations/10) == 0 or iteration + 1 == max_iterations:
            print(f'{int(100*iteration/max_iterations)}% of iterations used')

        drift = numpy.zeros(positions.shape)

        for i in range(particle_number):

            x_i = positions[i]

            # for all particles j ...
            for j in range(particle_number):

                if i == j:
                    continue

                # ... and their periodic images ...
                for pbc_image in pbc_iterator(spatial_dimensions):

                    # PBC shifted x_j
                    x_j = positions[j] + box_sidelength * numpy.array(pbc_image)

                    # separation vector connecting j with i
                    separation = x_i - x_j

                    # prevent expensive operations by making a ...
                    # ... cheaper check
                    if any(separation > particle_diameter):
                        continue

                    separation_length = numpy.linalg.norm(separation)
                    normalized_separation = separation / separation_length

                    # if the particles overlap let the difference contribute ...
                    # ... to their drift
                    if separation_length <= particle_diameter:
                        drift[i] += normalized_separation

            if numpy.linalg.norm(drift[i]) > 0:
                drift[i] = drift_speed * (drift[i] / numpy.linalg.norm(drift[i]))

        # if there is no drift, then one of two things happened:
        #   (a) the system is in perfect mechnaical equilibrium,
        #       which is however very unlikely (Lebegue-zero in
        #       a continuous system)
        #   (b) the system is in an absorbing state, i.e. all
        #       overlaps have been successfully resolved
        if not numpy.linalg.norm(drift) > 0:
            return positions

        positions = (positions + drift) % box_sidelength

        # NOTE: temporary to watch the spheres arrange themselves, because
        #       that is interesting to watch.
        if xyz_dump_filepath:
            xyz_dump(
                xyz_dump_filepath, # 'overlap_reduction.xyz',
                ['ball']*particle_number,
                positions,
                radius=particle_diameter/2
            )

    warnings.warn(
        f'Could not find overlap free configurations in {max_iterations}'
        + f' step{'s' if max_iterations > 0 else ''}'
    )
    return positions


def sum_zero_unit_vectors(
        number: int,
        spatial_dimensions: int,
        max_iterations:int=int(1e3),
        step_size:float=1e-2,
        threshold:float=1e-6
        ) -> numpy.ndarray:
    '''
    Returns {particle_number} {spatial_dimensions}-dimensional unit
    vectors that approximately sum to zero by minimizing a functional that
    penalizes the magnitude of their sum under the constraint that they be
    unit vectors

    Args:
        number:
            number of sum zero unit vectors
        spatial_dimensions:
            number of spatial dimensions
        max_iterations:
            maximum number of iterations for the employed gradient descent
        step_size:
            step size of the employed gradient descent
        threshold:
            threshold below which a unit vector sum is considered zero
    Returns:
        numpy.ndarray of unit vectors that approximately sum to zero
    '''

    # starting out with independently drawn unit vectors
    unit_vectors = numpy.array([
        vector / numpy.linalg.norm(vector)
        for vector in 2 * (numpy.random.rand(number, spatial_dimensions) - 1/2)
    ])

    # gradient of the functional
    def functional_gradient(n, lag):
        '''
        Args:
            n:
                numpy.ndarray containing the vectors presently
                optimized
            lag:
                list of the current values of all Lagrange
                multipliers

        Returns:
            the gradient for gradient descent
        '''
        return numpy.array([
            numpy.sum(n, axis=0) - lag[i]*n[i]
            for i in range(number)
        ])

    for iteration in range(max_iterations):

        # TODO: generalize progress bar, better user feedback
        if iteration % int(max_iterations/10) == 0 or iteration + 1 == max_iterations:
            print(f'{int(100*iteration/max_iterations)}% of iterations used')

        # compute the current lagrange parameters
        lagrange_parameters = [
            1 + numpy.sum([
                numpy.dot(unit_vectors[i], unit_vectors[j])
                for j in range(number)
                if j != i
            ])
            for i in range(number)
        ]

        gradient = functional_gradient(unit_vectors, lagrange_parameters)

        # below a length of {threshold} the unit_vector sum is considered ...
        # ... effectively zero ...
        if numpy.linalg.norm(numpy.sum(unit_vectors, axis=0)) < threshold:

            # ... and the descent is terminated
            return unit_vectors

        unit_vectors -= step_size * gradient

    # if even max iterations could not get the gradient below the ...
    # ... threshold, just return the unit vector
    warnings.warn(
        f'Could not get unit vector sum norm below {threshold} in {max_iterations}'
        + f' step{'s' if max_iterations > 1 else ''}'
    )
    return unit_vectors


def two_directions_sum_zero_unit_vectors(number, spatial_dimensions):
    '''
    Returns an arraw of unit vectors that point either in +x
    or -x direction and sum to zero (obviously impossible)
    for odd numbers of vectors

    Args:
        number:
            number of sum zero unit vectors
        spatial_dimensions:
            number of spatial dimensions
    '''
    # odd number would prevent the while loop from finishing
    if not number % 2 == 0:
        raise ValueError(f'number must be even, but {number} is odd')

    # directions that the particles point in: 1 stands for ...
    # ... +x and -1 stands for -x
    directions = 2*numpy.random.randint(2, size=number)-1

    # in case we did not randomly already make them sum to ...
    # ... zero, we need to pick some of the representatives ...
    # ... of the dominant direction and flip them
    if not numpy.sum(directions) == 0:

        # determining what direction has too much weight
        overweight_direction = 1 if numpy.sum(directions) > 0 else -1

        # now while the unit vectors do not sum to zero ...
        while not numpy.sum(directions) == 0:

            # ... flip
            index = numpy.random.choice(
                numpy.where(directions == overweight_direction)[0]
            )

            directions[index] *= -1

    return numpy.array([
        [direction, *[0 for _ in range(spatial_dimensions-1)]]
        for direction in directions
    ])


if __name__ == '__main__':

    test_unit_vectors = two_directions_sum_zero_unit_vectors(20, 2)
    print(numpy.sum(test_unit_vectors, axis=0))
