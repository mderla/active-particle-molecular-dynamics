'''
Contains physical computations

Mishael Derla, FAU Erlangen-Nürnberg, winter term 2024/2025
'''

from itertools import product
import numpy


# iterator for periodic boundaries
def pbc_iterator(spatial_dimensions, outermost_shell=1):
    '''
    returns a d-tuple, where d=spatial_dinemensions, containing all
    combinations of -1, 0 and 1 as entries. (0,0,0) indexes the simulation
    cell itself (1,0,0), (-1,0,1), (1,1,0), etc. index its periodic
    images

    Args:
        spatial_dimensions:
            the number of spatial dimensions, i.e. the length of the tuple
        outermost_shell:
            around the cell itself - with index (0,0) in d=2 - the relevant 
            PBC images wrap like shells. outermost_shell=1 for example would
            produce (-1,-1), (-1,0), (-1,1), ..., (1,0), (1,1), around which
            (-2,-2), (-2,-1), etc. with max(tuple)=2 will again wrap like a
            shell, i.e. outermost_shell = max(tuple)

    Returns:
        Iterator over PBC images
    '''
    return product(range(-outermost_shell,outermost_shell+1), repeat=spatial_dimensions)


def pair_force(
    sep,
    pair_force_magnitude: float,
    diameter: float
) -> numpy.ndarray:
    '''
    Computes the pair repulsion of two particles located at
    x and y respectively

    Args:
        sep:
            separation vector connecting the positions of the
            repulsing particles

    Returns:
        force acting on particle at x (= -force of particle at y)
    '''
    r = numpy.linalg.norm(sep)
    sep_normalized = sep/r
    return -pair_force_magnitude * sep_normalized * numpy.heaviside(diameter - r, 1/2)


def repulsion(
    positions: numpy.ndarray,
    box_sidelength: float,
    particle_diameter: float,
    pair_force_magnitude: float
) -> numpy.ndarray:
    '''
    Computes the repulsive forces between particles

    Args:
        positions:
            the positions of the particles
        box_sidelength:
            the sidelength of the periodic box

    Retrns:
        numpy.ndarray of shape positions.shape containing the forces
    '''

    # number of particles is positions.shape[0]
    particle_number = positions.shape[0]
    spatial_dimensions = positions.shape[1]

    # forces to be retuned
    forces = numpy.zeros(positions.shape)

    # for all particles ...
    for i in range(particle_number):

        # ... check all possible interaction parterns j < i ...
        # TODO: here: replace with a lookup
        for j in range(i):

            # ... and (assuming short-range interaction) the ...
            # ... closest PBC images ...
            for pbc_image in pbc_iterator(spatial_dimensions):

                # by introducing for every pair also a version ...
                # ... of it, that is separated by a lattice ...
                # ... vector that shifts any position into a ...
                # ... closest periodic image
                lattice_vector = box_sidelength * numpy.array(pbc_image)
                separation_ij = positions[j] - positions[i] + lattice_vector 
                # prevent expensive operations by making a ...
                # ... cheaper check
                if any(separation_ij > particle_diameter):
                    continue

                # computing the pair force as it acts on i ...
                pair_force_ij = pair_force(
                    separation_ij,
                    # other physics data
                    pair_force_magnitude,
                    particle_diameter
                )

                # ... and adding the computed pair force to both ...
                # ... interaction partners, where it gains a minus ...
                # ... sign F_ij = -F_ji ...
                forces[i] += pair_force_ij
                forces[j] += -pair_force_ij

    return forces


if __name__ == '__main__':

    # testing some things...
    print('\n'.join(f'{pbc_image}' for pbc_image in pbc_iterator(3, 2)))
