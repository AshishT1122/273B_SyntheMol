"""Contains classes representing chemical reactions, including reagents and products."""
import re
from functools import cache
from typing import Any, Callable, Optional, Union

from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.special import comb


Molecule = Union[str, Chem.Mol]  # Either a SMILES string or an RDKit Mol object


class CarbonChainChecker:
    """Checks whether a SMARTS match with two fragments contains a path
     from one fragment to the other with only non-aromatic C-C connections."""

    def __init__(self, smarts: str) -> None:
        """Initializes the carbon chain checker.

        Note: This is dependent on RDKit assigning atom indices in the order of the atoms in the SMARTS.

        :param smarts: A SMARTS string containing the query. Must contain precisely two fragments.
        """
        if sum(char == '.' for char in smarts) != 1:
            raise ValueError('SMARTS must contain precisely two fragments (separate by ".").')

        self.smarts = smarts

        first_smarts, second_smarts = smarts.split('.')
        first_mol, second_mol = Chem.MolFromSmarts(first_smarts), Chem.MolFromSmarts(second_smarts)
        first_num_atoms = first_mol.GetNumAtoms()

        # Get the indices of all the "*" atoms, i.e., the beginning of side chains
        self.start_atoms = {
            atom.GetIdx()
            for atom in first_mol.GetAtoms()
            if atom.GetAtomicNum() == 0
        }
        self.end_atoms = {
            atom.GetIdx() + first_num_atoms
            for atom in second_mol.GetAtoms()
            if atom.GetAtomicNum() == 0
        }

    def __call__(self, mol: Chem.Mol, matched_atoms: list[int]) -> bool:
        """Checks whether a molecule that has matched to the SMARTS query satisfies the carbon chain criterion.

        :param mol: The Mol object of the molecule that matches the SMARTS query.
        :param matched_atoms: A list of indices of atoms in the molecule that match the SMARTS query.
                              The ith element of this list is the index of the atom in the molecule
                              that matches the atom at index i in the SMARTS query.
                              (Note: This is technically an RDKit vect object, but it can be treated like a list.)
        :return: Whether the matched molecule satisfies the carbon chain criterion.
        """
        # Initialize a set of visited atoms
        visited_atoms = set(matched_atoms)

        # Get start and end indices in the molecule of the side chains that being with carbon atoms
        start_atom_indices = {
            atom_index for start_atom in self.start_atoms
            if mol.GetAtomWithIdx(atom_index := matched_atoms[start_atom]).GetAtomicNum() == 6
        }
        end_atom_indices = {
            atom_index for end_atom in self.end_atoms
            if mol.GetAtomWithIdx(atom_index := matched_atoms[end_atom]).GetAtomicNum() == 6
        }

        # If none of the side chains of a fragment begin with carbon, return False since there is no path of carbons
        if len(start_atom_indices) == 0 or len(end_atom_indices) == 0:
            return False

        # Loop over the atoms that begin side chains
        for start_atom_index in start_atom_indices:
            # Get the starting atom from its index
            atom = mol.GetAtomWithIdx(start_atom_index)

            # Iterate through the neighbors, checking for only non-aromatic C-C until reaching an end atom
            while True:
                # Get the neighboring atoms
                neighbors = atom.GetNeighbors()
                neighbor_h_count = sum(neighbor.GetAtomicNum() == 1 for neighbor in neighbors)

                # Check if this atom is carbon, non-aromatic, and all single bonds (i.e., four neighbors) with two Hs
                if atom.GetAtomicNum() != 6 or atom.GetIsAromatic() or len(neighbors) != 4 or neighbor_h_count != 2:
                    break

                # Check if we've reached an end atom and return True if so since we've satisfied the criterion
                if atom.GetIdx() in end_atom_indices:
                    return True

                # Add this atom to visited atoms
                visited_atoms.add(atom.GetIdx())

                # Move on to the next carbon atom in the chain (if there is one)
                next_atoms = [
                    neighbor
                    for neighbor in neighbors
                    if neighbor.GetIdx() not in visited_atoms and neighbor.GetAtomicNum() == 6
                ]

                if len(next_atoms) != 1:
                    break

                atom = next_atoms[0]

        # If we get here, then there is no path that satisfies the carbon chain criterion so return False
        return False

    def __hash__(self) -> int:
        return hash(self.smarts)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CarbonChainChecker) and self.smarts == other.smarts


def count_one_reagent(num_r1: int) -> int:
    """Counts the number of feasible molecules created from one reagent.

    :param num_r1: The number of molecules that match the first reagent.
    :return: The number of different molecules that can be constructed using this reaction and the given reagents.
    """
    return num_r1


# TODO: document/remove diff
def count_two_different_reagents(num_r1: int, num_r2: int, diff: bool = False) -> int:
    """Counts the number of feasible molecules created from two different reagents.

    :param num_r1: The number of molecules that match the first reagent.
    :param num_r2: The number of molecules that match the second reagent.
    :return: The number of different molecules that can be constructed using this reaction and the given reagents.
    """
    return num_r1 * num_r2


# TODO: document/remove diff
def count_two_same_reagents(num_r1: int, num_r2: int, diff: bool = False) -> int:
    """Counts the number of feasible molecules created from two of the same reagent.

    :param num_r1: The number of molecules that match the first reagent.
    :param num_r2: The number of molecules that match the second reagent (this should be the same as num_r1).
    :return: The number of different molecules that can be constructed using this reaction and the given reagents.
    """
    if diff:
        return num_r1 * num_r2

    assert num_r1 == num_r2

    return comb(num_r1, 2, exact=True, repetition=True)


# TODO: document/remove diff
def count_three_reagents_with_two_same(num_r1: int, num_r2: int, num_r3: int, diff: bool = False) -> int:
    """Counts the number of feasible molecules created from three reagents
       with the last two the same and the first different.

    :param num_r1: The number of molecules that match the first reagent.
    :param num_r2: The number of molecules that match the second reagent.
    :param num_r3: The number of molecules that match the third reagent (this should be the same as num_r2).
    :return: The number of different molecules that can be constructed using this reaction and the given reagents.
    """
    if diff:
        return num_r1 * num_r2 * num_r3

    assert num_r2 == num_r3

    return num_r1 * comb(num_r2, 2, exact=True, repetition=True)


def strip_atom_mapping(smarts: str) -> str:
    """Strips the atom mapping from a SMARTS (i.e., any ":" followed by digits).

    :param smarts: A SMARTS string with atom mapping indices.
    :return: The same SMARTS string but without the atom mapping indices.
    """
    return re.sub(r'\[([^:]+)(:\d+)]', r'[\1]', smarts)


def convert_to_mol(mol: Molecule, add_hs: bool = False) -> Chem.Mol:
    """Converts a SMILES to an RDKit Mol object (if not already converted) and optionally adds Hs.

    :param mol: A SMILES string or an RDKit Mol object.
    :param add_hs: Whether to add Hs.
    :return: An RDKit Mol object with Hs added optionally.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    if add_hs:
        mol = Chem.AddHs(mol)

    return mol


class QueryMol:
    """Contains a molecule query in the form of a SMARTS string with helper functions."""

    def __init__(self,
                 smarts: str,
                 checker_class: Optional[type] = None) -> None:
        """Initializes the QueryMol.

        :param smarts: A SMARTS string representing the molecular query.
        :param checker_class: A class that is used as an extra checker for any molecules that match the SMARTS query.
        """
        self.smarts_with_atom_mapping = smarts if ':' in smarts else None
        self.smarts = strip_atom_mapping(smarts)
        self.query_mol = Chem.MolFromSmarts(self.smarts)

        self.params = Chem.SubstructMatchParameters()

        if checker_class is not None:
            self.checker = checker_class(self.smarts)
            self.params.setExtraFinalCheck(self.checker)
        else:
            self.checker = None

    # TODO: cache this (might need to switch to only using strings)
    @cache
    def has_substruct_match(self, mol: Molecule) -> bool:
        """Determines whether the provided molecule includes this QueryMol as a substructure.

        :param mol: A molecule, which can either be a SMILES string or an RDKit Mol object.
        :return: True if the molecule includes this QueryMol as a substructure, False otherwise.
        """
        mol = convert_to_mol(mol, add_hs=True)

        return mol.HasSubstructMatch(self.query_mol, self.params)

    def __hash__(self) -> int:
        """Gets the hash of the QueryMol. Note: The hash depends on the SMARTS *without* atom mapping."""
        return hash(self.smarts)

    def __eq__(self, other: Any) -> bool:
        """Determines equality with another object. Note: The equality depends on the SMARTS *without* atom mapping."""
        return isinstance(other, QueryMol) and self.smarts == other.smarts and self.checker == other.checker


class Reaction:
    """Contains a chemical reaction including SMARTS for the reagents, product, and reaction and helper functions."""

    def __init__(self,
                 reagents: list[QueryMol],
                 product: QueryMol,
                 reaction_id: Optional[int] = None,
                 real_ids: Optional[set[int]] = None,
                 synnet_ids: Optional[set[int]] = None,
                 counting_fn: Optional[Union[Callable[[int, int], int],
                                             Callable[[int, int, int], int]]] = None) -> None:
        """Initializes the Reaction.

        :param reagents: A list of QueryMols containing the reagents of the reaction.
        :param product: A QueryMol containing the product of the reaction.
        :param reaction_id: The ID of the reaction.
        :param real_ids: A set of reaction IDs from the REAL database that this reaction corresponds to.
        :param synnet_ids: A set of IDs from the SynNet database that this reaction corresponds to.
        :param counting_fn: A function that takes in the number of molecules that match each possible reagent
                            and outputs the number of possible product molecules.
        """
        self.reagents = reagents
        self.product = product
        self.id = reaction_id
        self.real_ids = real_ids
        self.synnet_id = synnet_ids
        self.counting_fn = counting_fn

        self.reaction = AllChem.ReactionFromSmarts(
            f'{".".join(f"({reagent.smarts_with_atom_mapping})" for reagent in self.reagents)}'
            f'>>({self.product.smarts_with_atom_mapping})'
        )

    @property
    def num_reagents(self) -> int:
        return len(self.reagents)

    def run_reactants(self, reactants: list[Molecule]) -> tuple[tuple[Chem.Mol, ...], ...]:
        return self.reaction.RunReactants([convert_to_mol(reactant, add_hs=True) for reactant in reactants])

    # TODO: document/remove diff
    def count_feasible_products(self, *num_rs: tuple[int], diff: bool = False) -> int:
        """Counts the number of feasible products of this reaction given the number of each reagent.

        :param num_rs: The number of different molecules that match each reagent in the reaction.
        :return: The number of feasible product molecules this reaction could produce given the number of reagents.
        """
        if self.counting_fn is None:
            raise ValueError('Counting function is not provided for this reaction.')

        return self.counting_fn(*num_rs, diff=diff)
