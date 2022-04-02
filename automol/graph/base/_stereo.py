""" graph functions that depend on stereo assignments

BEFORE ADDING ANYTHING, SEE IMPORT HIERARCHY IN __init__.py!!!!
"""

import functools
import itertools
import numpy
from automol import util
import automol.geom.base    # !!!!
from automol.util import dict_
from automol.graph.base._core import atom_keys
from automol.graph.base._core import atom_stereo_parities
from automol.graph.base._core import bond_stereo_parities
from automol.graph.base._core import set_atom_stereo_parities
from automol.graph.base._core import set_bond_stereo_parities
from automol.graph.base._core import frozen
from automol.graph.base._core import has_stereo
from automol.graph.base._core import without_stereo_parities
from automol.graph.base._core import atoms_neighbor_atom_keys
from automol.graph.base._algo import rings_atom_keys
from automol.graph.base._algo import branch
from automol.graph.base._canon import stereogenic_atom_keys
from automol.graph.base._canon import stereogenic_bond_keys
from automol.graph.base._canon import to_local_stereo
from automol.graph.base._canon import atom_parity_evaluator_from_geometry_
from automol.graph.base._canon import bond_parity_evaluator_from_geometry_


# # core functions
def stereomers(gra):
    """ all stereomers, ignoring this graph's assignments
    """
    bool_vals = (False, True)

    def _expand_atom_stereo(sgr):
        atm_ste_keys = stereogenic_atom_keys(sgr)
        nste_atms = len(atm_ste_keys)
        sgrs = [set_atom_stereo_parities(sgr, dict(zip(atm_ste_keys,
                                                       atm_ste_par_vals)))
                for atm_ste_par_vals
                in itertools.product(bool_vals, repeat=nste_atms)]
        return sgrs

    def _expand_bond_stereo(sgr):
        bnd_ste_keys = stereogenic_bond_keys(sgr)
        nste_bnds = len(bnd_ste_keys)
        sgrs = [set_bond_stereo_parities(sgr, dict(zip(bnd_ste_keys,
                                                       bnd_ste_par_vals)))
                for bnd_ste_par_vals
                in itertools.product(bool_vals, repeat=nste_bnds)]
        return sgrs

    last_sgrs = []
    sgrs = [without_stereo_parities(gra)]

    while sgrs != last_sgrs:
        last_sgrs = sgrs
        sgrs = list(itertools.chain(*map(_expand_atom_stereo, sgrs)))
        sgrs = list(itertools.chain(*map(_expand_bond_stereo, sgrs)))

    return tuple(sorted(sgrs, key=frozen))


def substereomers(gra):
    """ all stereomers compatible with this graph's assignments
    """
    _assigned = functools.partial(
        dict_.filter_by_value, func=lambda x: x is not None)

    known_atm_ste_par_dct = _assigned(atom_stereo_parities(gra))
    known_bnd_ste_par_dct = _assigned(bond_stereo_parities(gra))

    def _is_compatible(sgr):
        atm_ste_par_dct = _assigned(atom_stereo_parities(sgr))
        bnd_ste_par_dct = _assigned(bond_stereo_parities(sgr))
        _compat_atm_assgns = (set(known_atm_ste_par_dct.items()) <=
                              set(atm_ste_par_dct.items()))
        _compat_bnd_assgns = (set(known_bnd_ste_par_dct.items()) <=
                              set(bnd_ste_par_dct.items()))
        return _compat_atm_assgns and _compat_bnd_assgns

    sgrs = tuple(filter(_is_compatible, stereomers(gra)))
    return sgrs


# # stereo evaluation
def local_atom_stereo_parity_from_geometry(gra, atm_key, geo,
                                           geo_idx_dct=None):
    """ Determine the local stereo parity of an atom from a geometry

        Local stereo parities are given relative to the indices of neighboring
        atoms.

        :param gra: molecular graph with stereo parities
        :type gra: automol graph data structure
        :param atm_key: the atom whose parity is to be determined
        :type atm_key: int
        :param geo: molecular geometry
        :type geo: automol geometry data structure
        :param geo_idx_dct: If they don't already match, specify which graph
            keys correspond to which geometry indices.
        :type geo_idx_dct: dict[int: int]
    """
    atm_keys = atom_keys(gra)
    atm_par_eval_ = atom_parity_evaluator_from_geometry_(
        gra, geo, geo_idx_dct=geo_idx_dct)
    atm_par_ = atm_par_eval_(dict(zip(atm_keys, atm_keys)))
    return atm_par_(atm_key)


def local_bond_stereo_parity_from_geometry(gra, bnd_key, geo,
                                           geo_idx_dct=None):
    """ Determine the local stereo parity of a bond from a geometry

        Local stereo parities are given relative to the indices of neighboring
        atoms.

        :param gra: molecular graph with stereo parities
        :type gra: automol graph data structure
        :param bnd_key: the bond whose parity is to be determined
        :type bnd_key: int
        :param geo: molecular geometry
        :type geo: automol geometry data structure
        :param geo_idx_dct: If they don't already match, specify which graph
            keys correspond to which geometry indices.
        :type geo_idx_dct: dict[int: int]
    """
    atm_keys = atom_keys(gra)
    bnd_par_eval_ = bond_parity_evaluator_from_geometry_(
        gra, geo, geo_idx_dct=geo_idx_dct)
    bnd_par_ = bnd_par_eval_(dict(zip(atm_keys, atm_keys)))
    return bnd_par_(bnd_key)


# # stereo correction
def stereo_corrected_geometry(gra, geo, geo_idx_dct=None, local_stereo=False):
    """ Obtain a geometry corrected for stereo parities based on a graph

        :param gra: molecular graph with stereo parities
        :type gra: automol graph data structure
        :param geo: molecular geometry
        :type geo: automol geometry data structure
        :param geo_idx_dct: If they don't already match, specify which graph
            keys correspond to which geometry indices.
        :type geo_idx_dct: dict[int: int]
        :param local_stereo: is this graph using local instead of canonical
            stereo?
        :type local_stereo: bool
        :returns: a molecular geometry with corrected stereo
    """
    sgr = gra if local_stereo else to_local_stereo(gra)
    gra = without_stereo_parities(gra)

    if has_stereo(sgr):
        full_atm_par_dct = atom_stereo_parities(sgr)
        full_bnd_par_dct = bond_stereo_parities(sgr)

        atm_keys = set()
        bnd_keys = set()

        last_gra = None

        while last_gra != gra:
            last_gra = gra

            atm_keys.update(stereogenic_atom_keys(gra))
            bnd_keys.update(stereogenic_bond_keys(gra))

            atm_par_dct = {k: full_atm_par_dct[k] for k in atm_keys}
            bnd_par_dct = {k: full_bnd_par_dct[k] for k in bnd_keys}
            geo, gra = _local_atom_stereo_corrected_geometry(
                gra, atm_par_dct, geo, geo_idx_dct)
            geo, gra = _local_bond_stereo_corrected_geometry(
                gra, bnd_par_dct, geo, geo_idx_dct)

    return geo


def _local_atom_stereo_corrected_geometry(gra, atm_par_dct, geo,
                                          geo_idx_dct=None):
    """ Correct a geometry to match local atom stereo assignments.

        :param gra: molecular graph
        :type gra: automol graph data structure
        :param atm_par_dct: local atom parities (local means relative to the
            neighboring atom keys)
        :type atm_par_dct: dict
        :param geo: molecular geometry
        :type geo: automol geometry data structure
        :param geo_idx_dct: If they don't already match, specify which graph
            keys correspond to which geometry indices.
        :type geo_idx_dct: dict[int: int]
    """
    atm_keys = atom_keys(gra)
    ring_atm_keys = set(itertools.chain(*rings_atom_keys(gra)))
    atm_ngb_keys_dct = atoms_neighbor_atom_keys(gra)

    geo_idx_dct = (geo_idx_dct if geo_idx_dct is not None
                   else {k: i for i, k in enumerate(atm_keys)})

    # Create a parity evaluator
    atm_par_eval_ = atom_parity_evaluator_from_geometry_(
        gra, geo, geo_idx_dct=geo_idx_dct)
    atm_par_ = atm_par_eval_(dict(zip(atm_keys, atm_keys)))

    ste_atm_keys = list(atm_par_dct.keys())
    for atm_key in ste_atm_keys:
        par = atm_par_dct[atm_key]
        curr_par = atm_par_(atm_key)

        if curr_par != par:
            atm_ngb_keys = atm_ngb_keys_dct[atm_key]
            # for now, we simply exclude rings from the pivot keys
            # (will not work for stereo atom at the intersection of two rings)
            atm_piv_keys = list(atm_ngb_keys - ring_atm_keys)[:2]
            assert len(atm_piv_keys) == 2
            atm3_key, atm4_key = atm_piv_keys

            # get coordinates
            xyzs = automol.geom.base.coordinates(geo)
            atm_xyz = xyzs[geo_idx_dct[atm_key]]
            atm3_xyz = xyzs[geo_idx_dct[atm3_key]]
            atm4_xyz = xyzs[geo_idx_dct[atm4_key]]

            # do the rotation
            rot_axis = util.vec.unit_bisector(
                atm3_xyz, atm4_xyz, orig_xyz=atm_xyz)

            rot_atm_keys = (
                atom_keys(branch(gra, atm_key, {atm_key, atm3_key})) |
                atom_keys(branch(gra, atm_key, {atm_key, atm4_key})))
            rot_idxs = list(map(geo_idx_dct.__getitem__, rot_atm_keys))

            geo = automol.geom.rotate(
                geo, rot_axis, numpy.pi, orig_xyz=atm_xyz, idxs=rot_idxs)

        gra = set_atom_stereo_parities(gra, {atm_key: par})

    return geo, gra


def _local_bond_stereo_corrected_geometry(gra, bnd_par_dct, geo,
                                          geo_idx_dct=None):
    """ Correct a geometry to match local bond stereo assignments.

        :param gra: molecular graph
        :type gra: automol graph data structure
        :param bnd_par_dct: local bond parities (local means relative to the
            neighboring atom keys)
        :type bnd_par_dct: dict
        :param geo: molecular geometry
        :type geo: automol geometry data structure
        :param geo_idx_dct: If they don't already match, specify which graph
            keys correspond to which geometry indices.
        :type geo_idx_dct: dict[int: int]
    """
    atm_keys = atom_keys(gra)
    bnd_keys = list(bnd_par_dct.keys())

    geo_idx_dct = (geo_idx_dct if geo_idx_dct is not None
                   else {k: i for i, k in enumerate(atm_keys)})

    # Create a parity evaluator
    bnd_par_eval_ = bond_parity_evaluator_from_geometry_(
        gra, geo, geo_idx_dct=geo_idx_dct)
    bnd_par_ = bnd_par_eval_(dict(zip(atm_keys, atm_keys)))

    for bnd_key in bnd_keys:
        par = bnd_par_dct[bnd_key]
        curr_par = bnd_par_(bnd_key)

        if curr_par != par:
            xyzs = automol.geom.base.coordinates(geo)

            atm1_key, atm2_key = bnd_key
            atm1_xyz = xyzs[geo_idx_dct[atm1_key]]
            atm2_xyz = xyzs[geo_idx_dct[atm2_key]]

            rot_axis = numpy.subtract(atm2_xyz, atm1_xyz)

            rot_atm_keys = atom_keys(
                branch(gra, atm1_key, {atm1_key, atm2_key}))

            rot_idxs = list(map(geo_idx_dct.__getitem__, rot_atm_keys))

            geo = automol.geom.rotate(
                geo, rot_axis, numpy.pi, orig_xyz=atm1_xyz, idxs=rot_idxs)

        gra = set_bond_stereo_parities(gra, {bnd_key: par})

    return geo, gra
