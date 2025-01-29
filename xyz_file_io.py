'''
IO for xyz files

Mishael Derla, FAU Erlangen-Nürnberg, winter term 2024/2025
'''

from dataclasses import dataclass
from typing import Iterable
import numpy


def xyz_dump(
        file_path: str,
        types: list[str],
        positions: numpy.ndarray,
        propulsions:numpy.ndarray=None,
        comment:str='',
        radius:str=''
        ) -> None:
    '''
    Dumps positions to an XYZ file

    Args:
        file_path:
            file path of the dump
        positions:
            numpy.ndarray of shape (N,d) where N is the number
            of particles and d the number of spatial dimensions
        comment:
            the XYZ comment line

    Returns:
        None
    '''

    if '\n' in comment:
        raise ValueError(
            'Comment line cannot contain linebreaks, lest XYZ file format be messed up'
        )

    # number of particles is positions.shape[0]
    N = positions.shape[0]

    with open(file_path, 'a', encoding='ascii') as file:

        # header: in XYZ file format (https://en.wikipedia.org/wiki/XYZ_file_format) ...
        # ... one needs to first speficy the particle number
        print(f'{N}', file=file)

        # next line needs to be at least clear, but can contain a comment line
        print(comment, file=file)

        # finally the positions will be dumped
        for i in range(N):
            print(
                f'{types[i]} '
                + ' '.join(str(component) for component in positions[i])
                + ' '
                + ' '.join(
                    (str(component) for component in propulsions[i])
                    if propulsions is not None else []
                )
                + f' {radius}',
                file=file
            )


@dataclass(frozen=True)
class XYZ:
    '''
    class whose properties reflect all information of an
    XYZ block. Note that the interpretation of the columns
    is up to the user

    Properties:
        particle_number:
            the number of particles in the block
        comment:
            a string containing the comment line
        labels:
            the particle labels
        xyz_data:
            the numerical XYZ-data columns

    Methods:
        is_valid:
            checks whether the XYZ-block is consistent

    Static Methods:
        file_iterator:
            returns an iterator over all XYZ-blocks in
            an XYZ-file at the passed filepath

    For XYZ file format reference, see:
        https://en.wikipedia.org/wiki/XYZ_file_format
    '''
    particle_number: int
    comment: str
    labels: list[str]
    xyz_data: numpy.ndarray


    def is_valid(self):
        '''
        Checks whether the XYZ block is valid
        '''
        label_count = len(self.labels)
        xyz_row_number = self.xyz_data.shape[0]

        return label_count == self.particle_number and xyz_row_number == self.particle_number


    @staticmethod
    def file_iterator(file_path: str) -> Iterable['XYZ']:
        '''
        Iterator that on each iteration returns a block of an
        XYZ file

        Args:
            file_path:
                path of the XYZ-file over whose blocks should
                be iterated

        Iterator.
        '''

        with open(file_path, 'r', encoding='ascii') as file:

            # while loop always causing another readline
            while line := file.readline():

                # we ensure that the line read in by the ...
                # ... while-loop is the particle number ...
                particle_number = int(line.strip())

                # ... by reading in the comment line ...
                comment = file.readline()

                # ... followed by a reliable number of lines
                labels = []
                xyz_data = []
                for _ in range(particle_number):

                    # splitting the label at the beginning of the line from the ...
                    # ... numerical information
                    particle_descriptor, *xyz_columns = file.readline().split(' ')

                    # appending them to separate lists
                    labels.append(particle_descriptor)
                    xyz_data.append([float(column) for column in xyz_columns])

                xyz_data = numpy.array(xyz_data)

                yield XYZ(
                    particle_number=particle_number,
                    comment=comment,
                    labels=labels,
                    xyz_data=xyz_data
                )


def active_particle_metadata(filepath: str):
    '''
    Grabs and parses the metadata string we stored into the
    comment field of our .xyz files
    
    Args:
		filepath:
			Path of the .xyz file
    
    Returns:
		a dictionary with strings containing the physical
        property name as keys and the corresponding numerical
        values as values
    '''
    with open(filepath, 'r', encoding='ascii') as file:

		# first line contains the particle number, thus skip
        next(file)

		# read in comment string (will be the same for all XYZ ...
		# ... blocks, so we need to only read until here)
        xyz_comment = file.readline()

		# parse
        return {
			statement.split('=')[0] : float(statement.split('=')[1])
			for statement in xyz_comment.split(';')[1].replace(' ', '').split(',')
		}
