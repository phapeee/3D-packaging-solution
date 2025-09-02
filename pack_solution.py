"""
pack_solution.py
===================

This module implements a simple packaging helper on top of the 3D bin
packing library contained in the GitHub repository
``jerry800416/3D‑bin‑packing``.  The goal of the helper is to decide
whether a set of items can fit into one of a list of padded mailers
before falling back to a three‑dimensional box.  When a suitable
mailer or box is found, the function returns a dimension string
(``"WxL"`` for mailers or ``"WxLxH"`` for boxes).  When debug
mode is enabled the helper also produces interactive 3D
visualisations: one showing each distinct item on its own and a
second showing the chosen container with all items placed inside.

**Usage overview**
-----------------

The main entry point is :func:`choose_container`.  It accepts three
lists describing the available padded mailers, the available boxes and
the items to pack.  Each mailer is given as a string of the form
``"widthxlength"`` (in inches) and each box is given as a string of
the form ``"widthxheightxdepth"``.  Items are dictionaries containing
an ``id`` (a string identifier), a ``dimension`` (for example
``"1x2x3"``) and a ``quantity`` (an integer).  The function first
attempts to pack all items into each mailer (starting with the
smallest by area) using a simple two‑dimensional shelf algorithm.  A
mailer is treated as having a fixed height of four inches and items
may not be stacked on top of one another.  If no mailer can
accommodate the items, the helper falls back to three‑dimensional
packing using the ``py3dbp`` classes.  For details on how to create
bins, items and pack them the reader is referred to the `How to use`
section of the upstream README【170516555184532†L417-L472】.

The code has no external dependencies beyond ``numpy``,
``matplotlib`` and the ``py3dbp`` package; see ``requirements.txt`` in
the repository for version information【170516555184532†L417-L472】.  The
``py3dbp`` package itself is imported from the repository folder.

"""

# from __future__ import annotations

import itertools
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

layer_tol = 0.02

# Inject the cloned repository on to the Python search path so that
# py3dbp can be imported.  When this module is used from a different
# location you may need to adjust the relative path below.
REPO_PATH = os.path.join(
    os.path.dirname(__file__), "3d_bin_pack_repo", "3D-bin-packing-master"
)
if os.path.isdir(REPO_PATH) and REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

try:
    # The core classes required for 3D packing.  According to the
    # upstream README, packing requires creating a :class:`Bin` and
    # :class:`Item` objects, then adding them to a :class:`Packer` and
    # finally calling :meth:`Packer.pack`【170516555184532†L417-L472】.
    from py3dbp import Bin, Item, Packer, Painter  # type: ignore
except ImportError:  # pragma: no cover - runtime import error only
    raise ImportError(
        "Unable to import py3dbp.  Make sure the Git repository has been "
        "downloaded and that REPO_PATH points to the root of the repo."
    )


@dataclass
class ItemInstance:
    """Represents a single instance of an item.

    The original item list may contain a ``quantity`` field; each
    occurrence of the item is represented as a separate
    :class:`ItemInstance` so that it can be tracked during packing.
    ``dims`` holds the three edges of the cuboid in ascending order.
    """

    id: str
    dims: Tuple[float, float, float]


def parse_dims(dim_str: str) -> Tuple[int, ...]:
    """Parse a dimension string of the form ``"a x b x c"`` into a tuple.

    Parameters
    ----------
    dim_str: str
        Dimension string (case insensitive) using ``x`` as a separator.

    Returns
    -------
    Tuple[int, ...]
        The dimensions as integers in the order encountered in the
        string.  Spaces are ignored.
    """
    parts = dim_str.lower().replace(" ", "").split("x")
    if not parts or any(not p.isdigit() for p in parts):
        raise ValueError(f"Invalid dimension string: {dim_str}")
    return tuple(int(p) for p in parts)

def parse_dims_float(dim_str: str) -> Tuple[float, ...]:
    """Parse a dimension string of the form ``"a x b x c"`` into a tuple of floats.

    Parameters
    ----------
    dim_str : str
        Dimension string (case insensitive) using ``x`` as a separator.

    Returns
    -------
    Tuple[float, ...]
        The dimensions as floats in the order encountered in the
        string. Spaces are ignored.
    """
    parts = dim_str.lower().replace(" ", "").split("x")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise ValueError(f"Invalid dimension string: {dim_str}")


def sort_mailers(mailers: List[str]) -> List[Tuple[int, int, str]]:
    """Sort mailers by area in ascending order.

    Each mailer string is parsed into a pair of integers ``(width,
    length)``.  Because padded mailers are effectively 2D, the
    orientation doesn't matter; the smaller dimension is always taken
    as the width and the larger as the length.  The returned list
    contains tuples ``(width, length, original_string)`` sorted by
    ``width * length`` so that the smallest envelope is tried first.

    Parameters
    ----------
    mailers: List[str]
        List of mailer dimension strings, e.g. ``['10x8', '6x8']``.

    Returns
    -------
    List[Tuple[int, int, str]]
        Parsed mailers sorted by area.
    """
    parsed: List[Tuple[int, int, str]] = []
    for m in mailers:
        dims = parse_dims(m)
        if len(dims) != 2:
            raise ValueError(f"Mailers must have two dimensions, got {m}")
        w, l = dims
        # Enforce width <= length for canonical representation.
        width, length = (w, l) if w <= l else (l, w)
        parsed.append((width, length, m))
    # sort by area then by width
    parsed.sort(key=lambda x: (x[0] * x[1], x[0], x[1]))
    return parsed


def sort_boxes(boxes: List[str]) -> List[Tuple[int, int, int, str]]:
    """Sort boxes by volume in ascending order.

    Each box string is parsed into a triple of integers ``(width,
    height, depth)``.  The returned list contains tuples
    ``(w, h, d, original_string)`` sorted by volume so that the
    smallest box is tried first.

    Parameters
    ----------
    boxes: List[str]
        List of box dimension strings, e.g. ``['8x8x8', '10x10x10']``.

    Returns
    -------
    List[Tuple[int, int, int, str]]
        Parsed boxes sorted by volume.
    """
    parsed: List[Tuple[int, int, int, str]] = []
    for b in boxes:
        dims = parse_dims(b)
        if len(dims) != 3:
            raise ValueError(f"Boxes must have three dimensions, got {b}")
        parsed.append((*dims, b))
    # sort by volume then by dimensions
    parsed.sort(key=lambda x: (x[0] * x[1] * x[2], x[0], x[1], x[2]))
    return parsed


def generate_item_instances(items: List[Dict[str, Any]]) -> List[ItemInstance]:
    """Expand the input item list into individual instances.

    Each dictionary in ``items`` may specify a quantity greater than one;
    this helper duplicates the entry accordingly and stores the
    dimension tuple in ascending order.  The original ``id`` is
    preserved so that multiple instances of the same identifier can be
    distinguished (only by position in the list).

    Parameters
    ----------
    items: List[Dict[str, Any]]
        The user supplied items where each entry has keys ``id``,
        ``dimension`` and ``quantity``.

    Returns
    -------
    List[ItemInstance]
        A list of expanded :class:`ItemInstance` objects.
    """
    instances: List[ItemInstance] = []
    for itm in items:
        dims_raw = parse_dims_float(itm["dimension"])
        if len(dims_raw) != 3:
            raise ValueError(f"Items must have three dimensions, got {itm}")
        # store dimensions in ascending order for easier orientation
        dims_sorted = tuple(sorted(dims_raw))  # type: ignore
        for _ in range(itm.get("quantity", 1)):
            instances.append(ItemInstance(id=itm["id"], dims=dims_sorted))
    return instances

def can_fit_in_mailer(
    items: List[ItemInstance],
    mailer_width: float,
    mailer_length: float,
    flat_height: float
) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    """
    Attempt to pack items into a padded mailer allowing stacking (layering in z).

    - X = width, Y = length of the mailer footprint.
    - Items' smallest dimension (dims[0]) is their thickness (z).
    - We pack items into rows (shelves) within a layer (z = constant).
    - When no more room in current layer, start a new layer if total height allows.
    - A layer's height equals the max thickness of the items placed in that layer.
    """
    # All items must individually fit within the mailer's height
    for inst in items:
        if inst.dims[0] > flat_height:
            return False, None

    # Sort by descending face area (largest footprint first), then max side
    items_sorted = sorted(
        items, key=lambda it: (it.dims[2] * it.dims[1], it.dims[2]), reverse=True
    )

    placements: List[Dict[str, Any]] = []

    # Current layer bookkeeping
    z_offset: float = 0.0          # where this layer starts
    layer_height: float = 0.0      # max thickness in this layer

    # Within-layer shelf packing
    y_offset: float = 0.0          # start of current row within the layer
    row_height: float = 0.0        # max Y of the current row
    row_width: float = 0.0         # used X within the current row

    def try_place_in_current_layer(inst: ItemInstance) -> Optional[Dict[str, Any]]:
        """Try place 'inst' in current row, else new row, within this layer."""
        nonlocal row_width, row_height, y_offset

        t = inst.dims[0]  # thickness (z)
        # orientation 0: width=dims[2], length=dims[1]; orientation 1 swapped
        orientations = [
            (inst.dims[2], inst.dims[1], 0),
            (inst.dims[1], inst.dims[2], 1),
        ]

        # Try current row
        for w_i, l_i, ori in orientations:
            if (
                row_width + w_i <= mailer_width
                and l_i <= (mailer_length - y_offset)
            ):
                pos = (row_width, y_offset)
                row_width += w_i
                row_height = max(row_height, l_i)
                return {
                    "item": inst,
                    "orientation": ori,
                    "position": pos,
                    "width": w_i,
                    "length": l_i,
                    "z": z_offset,
                    "height": t,
                }

        # Start a new row in this layer
        y_offset += row_height
        row_width = 0.0
        row_height = 0.0

        for w_i, l_i, ori in orientations:
            if w_i <= mailer_width and l_i <= (mailer_length - y_offset):
                pos = (row_width, y_offset)
                row_width += w_i
                row_height = max(row_height, l_i)
                return {
                    "item": inst,
                    "orientation": ori,
                    "position": pos,
                    "width": w_i,
                    "length": l_i,
                    "z": z_offset,
                    "height": t,
                }

        return None  # doesn't fit in this layer's X/Y

    for inst in items_sorted:
        t = inst.dims[0]  # thickness

        # If this is the very first item (or a new layer just began), initialize layer
        if layer_height == 0.0 and row_height == 0.0 and row_width == 0.0:
            # Ensure it can start a new (empty) layer height-wise
            if z_offset + t > flat_height:
                return False, None

        placed = try_place_in_current_layer(inst)

        if placed is not None:
            placements.append(placed)
            layer_height = max(layer_height, t)
            continue

        # Need a new layer: advance z by current layer height, reset XY shelf state
        if layer_height == 0.0:
            # No space in empty layer means item footprint exceeds mailer footprint
            return False, None

        z_offset += layer_height
        if z_offset + t > flat_height:
            return False, None  # not enough height to start a new layer

        # Reset layer XY
        y_offset = 0.0
        row_height = 0.0
        row_width = 0.0
        layer_height = 0.0  # will be set from items placed in this fresh layer

        placed = try_place_in_current_layer(inst)
        if placed is None:
            # Even a fresh layer can't fit footprint-wise
            return False, None

        placements.append(placed)
        layer_height = max(layer_height, t)

    return True, placements


def pack_in_box(
    items: List[ItemInstance], box_dims: Tuple[int, int, int]
) -> Tuple[bool, Any]:
    """Attempt to pack items into a single 3D box using py3dbp.

    The 3D bin packing library allows items to be rotated in all
    directions (subject to the ``updown`` flag).  Items are added to
    the packer along with the candidate box and the packing process is
    executed.  If the list of unfitted items is empty then the
    placement succeeded.

    Parameters
    ----------
    items: List[ItemInstance]
        List of individual items to be placed.
    box_dims: Tuple[int, int, int]
        The dimensions of the box as ``(width, height, depth)``.

    Returns
    -------
    Tuple[bool, Any]
        A boolean indicating whether the items all fitted and the
        resulting :class:`Packer` instance (to allow retrieval of
        placements on success).  On failure the second element is
        ``None``.
    """
    # Create a single bin for this box candidate.  The README shows
    # that a bin is initialised with a part number, a dimension tuple
    # and a maximum weight【170516555184532†L417-L472】.  The ``corner`` and
    # ``put_type`` parameters are optional and omitted here.
    width, height, depth = box_dims
    bin_obj = Bin(partno="candidate_box", WHD=(width, height, depth), max_weight=10**9)
    packer = Packer()
    packer.addBin(bin_obj)
    # Add each item with unique part numbers.  We set updown=True to
    # allow the item to be turned upside down if needed for better
    # packing.  Weight and load bearing are kept minimal for this
    # example.
    for idx, inst in enumerate(items):
        # Use the original sorted dims as the nominal (width, height, depth).
        # Because py3dbp will consider all rotations, the order here is
        # not critical; however, assigning the largest dimension as the
        # width gives the packer a reasonable starting orientation.
        dims = inst.dims
        # Rearrange dims so that the largest dimension comes first.
        dims_sorted_desc = tuple(sorted(dims, reverse=True))
        # Create the Item.  Colour is selected using a cyclic list of
        # colours for easy visualisation in debug mode.
        color = plt.cm.tab20(idx % 20)
        py_item = Item(
            partno=f"{inst.id}_{idx}",
            name=inst.id,
            typeof="cube",
            WHD=dims_sorted_desc,
            weight=1,
            level=1,
            loadbear=100,
            updown=True,
            color=color,
        )
        packer.addItem(py_item)
    # Perform the packing.  The README recommends using ``bigger_first``
    # and ``fix_point`` to obtain stable placements【170516555184532†L417-L472】.
    # Occasionally py3dbp may throw exceptions (e.g. due to
    # division‑by‑zero in gravity calculations when no items fit).  In
    # such cases we treat the attempt as a failure to fit.
    try:
        packer.pack(
            bigger_first=True,
            fix_point=True,
            distribute_items=True,
            # According to the README, enabling check_stable and setting
            # a support surface ratio can help the algorithm place
            # items properly【170516555184532†L417-L472】.  These values
            # are taken from the example code.
            check_stable=True,
            support_surface_ratio=0.75,
            number_of_decimals=1,
        )
        success = len(packer.unfit_items) == 0
    except Exception:
        success = False
        packer = None
    return success, packer if success else None



def pack_as_many_in_box(
    items: List[ItemInstance], box_dims: Tuple[int, int, int]
) -> Tuple[Any, List[ItemInstance]]:
    """
    Pack as many of 'items' as possible in a single box of 'box_dims'.
    Returns (packer, remaining_items). If packing throws an exception,
    returns (None, items) meaning nothing got packed.
    """
    width, height, depth = box_dims
    bin_obj = Bin(partno="candidate_box", WHD=(width, height, depth), max_weight=10**9)
    packer = Packer()
    packer.addBin(bin_obj)

    # Track back-references to ItemInstance objects
    py_items: List[Tuple[Any, ItemInstance]] = []
    for idx, inst in enumerate(items):
        dims = inst.dims
        dims_sorted_desc = tuple(sorted(dims, reverse=True))
        color = plt.cm.tab20(idx % 20)
        py_item = Item(
            partno=f"{inst.id}_{idx}",
            name=inst.id,
            typeof="cube",
            WHD=dims_sorted_desc,
            weight=1,
            level=1,
            loadbear=100,
            updown=True,
            color=color,
        )
        packer.addItem(py_item)
        py_items.append((py_item, inst))

    try:
        packer.pack(
            bigger_first=True,
            fix_point=True,
            distribute_items=True,
            check_stable=True,
            support_surface_ratio=0.75,
            number_of_decimals=1,
        )
    except Exception:
        return None, list(items)  # nothing packed

    # Determine which items didn't fit — compare by 'partno' instead of object identity
    unfit_partnos = {getattr(i, "partno", None) for i in getattr(packer, "unfit_items", [])}
    unfit_partnos.discard(None)

    # Fallback: if the library didn't populate unfit_items, infer unfit by subtracting placed from all
    if not unfit_partnos:
        placed_partnos = {
            getattr(it, "partno", None)
            for b in getattr(packer, "bins", [])
            for it in getattr(b, "items", [])
        }
        all_partnos = {py_item.partno for (py_item, _inst) in py_items}
        unfit_partnos = all_partnos - placed_partnos

    remaining: List[ItemInstance] = [
        inst for (py_item, inst) in py_items if py_item.partno in unfit_partnos
    ]
    return packer, remaining

def greedy_pack_into_biggest_box(
    items: List[ItemInstance],
    boxes: List[str],
    debug: bool = False,
    box_offset: int = 2,
) -> Tuple[str, Optional[List[Any]], List[ItemInstance], List[ItemInstance]]:
    """
    Use the largest box size available. Pack as many items as possible into
    that box; then recursively pack the remaining items into more copies of
    the same largest box until no items remain or we can't place anything.

    Returns:
        (box_dim_string_with_count,
         [list_of_packers] or None,
         leftover_items,
         last_box_items)   # <— NEW: the ItemInstance objects placed in the final box
    """
    sb = sort_boxes(boxes)
    if not sb:
        return "", None, list(items), []

    w, h, d, _bstr = sb[-1]
    base_perms = list(set(itertools.permutations(
        (w - box_offset, h - box_offset, d - box_offset)
    )))

    remaining = list(items)
    packers: List[Any] = []
    box_count = 0
    last_box_items: List[ItemInstance] = []  # track items packed in the most recent box

    while remaining:
        best_packer = None
        best_remaining = None
        best_unfit = len(remaining) + 1

        for perm in base_perms:
            packer, rem = pack_as_many_in_box(remaining, perm)
            if packer is None:
                continue
            if len(rem) < best_unfit:
                best_unfit = len(rem)
                best_packer = packer
                best_remaining = rem

        # No orientation could place anything → stop
        if best_packer is None or best_remaining is None:
            break

        # If no progress, stop to avoid infinite loop
        if len(best_remaining) == len(remaining):
            break

        # ---- Items that were actually placed in THIS box ----
        # Both 'remaining' and 'best_remaining' are lists of *your* ItemInstance objects,
        # so identity/equality is stable. Items that disappeared from 'remaining'
        # are exactly the ones packed this iteration.
        this_box_items = [inst for inst in remaining if inst not in best_remaining]
        last_box_items = this_box_items  # overwrite so we keep only the final box's contents

        if debug and best_packer is not None:
            visualise_box_solution(best_packer, (w, h, d), box_offset)

        packers.append(best_packer)
        box_count += 1
        remaining = best_remaining  # continue with leftovers

    # Canonical dimension string (sorted asc + add back offset)
    sorted_dims = sorted((w - box_offset, h - box_offset, d - box_offset))
    box_str = f"{sorted_dims[0] + box_offset}x{sorted_dims[1] + box_offset}x{sorted_dims[2] + box_offset}"
    if box_count == 0:
        return box_str + "*0", packers if debug else None, remaining, []
    if box_count > 1:
        box_count = box_count - 1
        packers = packers[:-1]
    else:
        last_box_items = []

    last_box_items = summarize_item_instances(last_box_items)

    return f"{box_str}*{box_count}", packers, remaining, last_box_items

def set_axes_equal(ax: plt.Axes) -> None:
    """Adjust a 3D axis so that all axes are scaled equally.

    Matplotlib does not natively support setting equal scales on
    3‑dimensional plots.  This helper ensures that cuboids and other
    shapes do not appear distorted by calculating the full extent of
    each axis and expanding the shorter ones to match the largest
    extent.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def draw_cuboid(
    ax: plt.Axes,
    origin: Tuple[float, float, float],
    size: Tuple[float, float, float],
    color: Any,
    label: Optional[str] = None,
) -> None:
    """Draw a solid cuboid on a 3D axis.

    Parameters
    ----------
    ax: plt.Axes
        A 3D axes instance returned by ``plt.subplot`` with
        ``projection='3d'``.
    origin: Tuple[float, float, float]
        The ``(x, y, z)`` coordinate of the lower left, bottom corner of
        the cuboid.
    size: Tuple[float, float, float]
        The size of the cuboid along each axis: ``(width, length,
        height)``.
    color: Any
        A valid matplotlib colour specification.  Colours may include
        alpha components.
    label: Optional[str]
        If provided, text is drawn at the centre of the cuboid using
        this label.
    """
    X, Y, Z = origin
    dx, dy, dz = size
    # Define the eight vertices of the cuboid.
    vertices = [
        (X, Y, Z),
        (X + dx, Y, Z),
        (X + dx, Y + dy, Z),
        (X, Y + dy, Z),
        (X, Y, Z + dz),
        (X + dx, Y, Z + dz),
        (X + dx, Y + dy, Z + dz),
        (X, Y + dy, Z + dz),
    ]
    # Define the six faces by listing the vertices that make up each face.
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]],
    ]
    poly = Poly3DCollection(
        faces, facecolors=[color], edgecolors="black", linewidths=1, alpha=0.6
    )
    ax.add_collection3d(poly)
    if label:
        ax.text(
            X + dx / 2,
            Y + dy / 2,
            Z + dz / 2,
            label,
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )


def _to_offset2(mailer_offset):
    """Accept a scalar or (ox, oy); return 2 floats."""
    if isinstance(mailer_offset, (int, float)):
        return float(mailer_offset), float(mailer_offset)
    if isinstance(mailer_offset, (list, tuple)) and len(mailer_offset) == 2:
        return float(mailer_offset[0]), float(mailer_offset[1])
    raise ValueError("mailer_offset must be a number or a 2-tuple/list")


def visualise_mailer_solution(
    mailer_dims: Tuple[int, int],
    placements: List[Dict[str, Any]],
    mailer_offset,  # scalar or (ox, oy); height has NO offset
    flat_height: int = 3,
) -> None:
    """
    Visualise a padded mailer packing solution.
    Draw the mailer as a shallow box at (0,0,0) with height 'flat_height',
    place items, then overlay an OUTER wireframe expanded only in X/Y by
    'mailer_offset' (no change to height).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    width, length = mailer_dims

    # Draw the base mailer (no offset in z).
    draw_cuboid(
        ax,
        (0, 0, 0),
        (width, length, flat_height),
        color=(0.95, 0.95, 0.95, 0.3),
        label=None,
    )

    # Place items
    cmap = plt.cm.get_cmap("tab20")
    for idx, placement in enumerate(placements):
        inst = placement["item"]
        if placement["orientation"] == 0:
            dx, dy = inst.dims[2], inst.dims[1]
        else:
            dx, dy = inst.dims[1], inst.dims[2]
        dz = inst.dims[0]
        x, y = placement["position"]
        color = cmap(idx % 20)
        z = float(placement.get("z", 0.0))
        draw_cuboid(ax, (x, y, z), (dx, dy, dz), color=color, label=inst.id)

    # ---- Outer wireframe (offset only in X & Y; height unchanged) ----
    ox, oy = _to_offset2(mailer_offset / 2)
    # Expand equally around all sides → shift origin negatively in x/y
    x0, y0, z0 = -ox, -oy, 0.0
    dx = float(width) + 2.0 * ox
    dy = float(length) + 2.0 * oy
    dz = float(flat_height)  # no z offset

    _draw_wire_cube(ax, x0, y0, z0, dx, dy, dz, linewidth=2)  # add color='k' if desired

    ax.set_title("Padded mailer packing solution")
    ax.set_xlabel("Width (x)")
    ax.set_ylabel("Length (y)")
    ax.set_zlabel("Height (z)")
    set_axes_equal(ax)

    try:
        plt.show()
    except Exception:
        pass  # safe in headless environments


def _to_offset3(box_offset):
    """Accept a scalar or (ox, oy, oz); return 3 floats."""
    if isinstance(box_offset, (int, float)):
        return float(box_offset), float(box_offset), float(box_offset)
    if isinstance(box_offset, (list, tuple)) and len(box_offset) == 3:
        return float(box_offset[0]), float(box_offset[1]), float(box_offset[2])
    raise ValueError("box_offset must be a number or a 3-tuple/list")


def _draw_wire_cube(ax, x, y, z, dx, dy, dz, **kwargs):
    """Draw a rectangular wireframe parallelepiped like Painter's mode=1."""
    xx = [x, x, x + dx, x + dx, x]
    yy = [y, y + dy, y + dy, y, y]
    ax.plot3D(xx, yy, [z] * 5, **kwargs)
    ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
    ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)


def visualise_box_solution(packer, box_dims, box_offset):
    """
    Draws the packed bin(s) using Painter, then overlays an enlarged outer
    wireframe box whose size = (width + 2*ox, height + 2*oy, depth + 2*oz).
    The inner bin starts at (0,0,0), so the outer box starts at (-ox,-oy,-oz)
    to wrap evenly around the inner one.
    """
    bins = iter_bins(packer)  # your compat helper from earlier
    last_fig = None

    for b in bins:
        painter = Painter(b)
        plt_obj = painter.plotBoxAndItems(
            title=b.partno, alpha=0.3, write_num=True, fontsize=8
        )

        # Get current 3D axes created by Painter
        ax = plt.gca()

        # Compute enlarged outer box
        ox, oy, oz = _to_offset3(box_offset / 2)
        x0, y0, z0 = -ox, -oy, -oz
        dx = float(b.width) + 2.0 * ox
        dy = float(b.height) + 2.0 * oy
        dz = float(b.depth) + 2.0 * oz

        # Draw outer wireframe
        _draw_wire_cube(ax, x0, y0, z0, dx, dy, dz, linewidth=2)

        last_fig = plt_obj

    if bins:
        try:
            last_fig.show()
        except Exception:
            pass  # headless environments


def visualise_individual_items(items: List[ItemInstance]) -> None:
    """Visualise each unique item individually in 3D.

    Items are arranged along the x axis with a small gap between them.
    Each cuboid is drawn using a distinct colour and labelled with
    the item identifier.  This view helps to understand the raw
    dimensions of each object before packing.

    Parameters
    ----------
    items: List[ItemInstance]
        List of item instances to visualise.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    offset = 0.0
    gap = 1.0  # extra space between items
    cmap = plt.cm.get_cmap("tab20")
    for idx, inst in enumerate(items):
        dx, dy, dz = (
            inst.dims[2],
            inst.dims[1],
            inst.dims[0],
        )  # largest to smallest along x,y,z
        color = cmap(idx % 20)
        draw_cuboid(ax, (offset, 0, 0), (dx, dy, dz), color=color, label=inst.id)
        offset += dx + gap
    ax.set_title("Individual items")
    ax.set_xlabel("Width (x)")
    ax.set_ylabel("Length (y)")
    ax.set_zlabel("Height (z)")
    set_axes_equal(ax)
    plt.show()


def mailer_arrangement_json(placements):
    out = []
    for p in placements or []:
        inst = p["item"]
        out.append(
            {
                "step": 1,
                "id": inst.id,
                "x": p["position"][0],
                "y": p["position"][1],
                "z": float(p.get("z", 0.0)),          # <-- new
                "w": p["width"],
                "l": p["length"],
                "h": float(p.get("height", inst.dims[0])),  # <-- new (thickness)
                "orientation": "x-long" if p["orientation"] == 0 else "y-long",
            }
        )
    return out

def _item_pos(item) -> Tuple[float, float, float]:
    pos = getattr(item, "position", (0, 0, 0))
    # Robust to tuple/list or shorter sequences
    x = float(pos[0]) if len(pos) > 0 else 0.0
    y = float(pos[1]) if len(pos) > 1 else 0.0
    z = float(pos[2]) if len(pos) > 2 else 0.0
    return x, y, z

def _item_dims_xyz(item) -> Tuple[float, float, float]:
    """
    Try to read oriented dimensions along the world axes (x,y,z) after packing.
    py3dbp typically exposes width/height/depth on placed items. If not, we
    fall back to getDimension() or WHD.
    We map:
        x-axis width  -> w
        y-axis depth  -> l
        z-axis height -> h
    """
    # Preferred: explicit oriented attributes
    w = getattr(item, "width", None)
    h = getattr(item, "height", None)
    d = getattr(item, "depth", None)
    if w is not None and h is not None and d is not None:
        return float(w), float(d), float(h)

    # Fallback: getDimension() (commonly returns (W, H, D))
    getdim = getattr(item, "getDimension", None)
    if callable(getdim):
        W, H, D = getdim()
        return float(W), float(D), float(H)

    # Fallback: original WHD (pre-rotation)
    WHD = getattr(item, "WHD", None)
    if WHD:
        W, H, D = WHD
        return float(W), float(D), float(H)

    # Last resort
    dims = getattr(item, "dims", (0, 0, 0))
    return float(dims[0]), float(dims[1]), float(dims[2])

def _cluster_layers_by_z(items: List[Any], tol: float) -> List[List[Any]]:
    """
    Group items into layers by their base Z (position.z), using a tolerance.
    Items whose z differ by <= tol belong to the same layer.
    """
    # Sort by base z ascending first
    items_sorted = sorted(items, key=lambda it: _item_pos(it)[2])
    layers: List[List[Any]] = []
    current: List[Any] = []
    last_z: float = None  # type: ignore

    for it in items_sorted:
        z = _item_pos(it)[2]
        if last_z is None or abs(z - last_z) <= tol:
            current.append(it)
        else:
            layers.append(current)
            current = [it]
        last_z = z

    if current:
        layers.append(current)
    return layers

def boxes_arrangement_json(
    packers: Union[Any, Iterable[Any]],
    layer_tol: float = 1e-3,
    within_layer_sort: str = "yx",  # "xy" or "yx": order items left-to-right & front-to-back
) -> List[Dict[str, Any]]:
    """
    Produce step-by-step placement instructions for boxes (bins) from py3dbp packers.
    - Steps are bottom-up Z layers.
    - All items in the same layer share the same 'step'.
    - Returns a flat list of dicts:
        {
          "box": <1-based index of the box across all packers>,
          "step": <1..N within that box>,
          "id": "<your id>",
          "x": <float>, "y": <float>, "z": <float>,
          "w": <float>, "l": <float>, "h": <float>,
          "orientation": "x-long" | "y-long",
        }
    Notes:
      * 'id' comes from item.name or item.partno (we set those when creating Items).
      * Orientation is derived by comparing the in-box X and Y extents.
    """
    # Normalize to an iterable of packers
    if not isinstance(packers, (list, tuple)):
        packers = [packers]

    out: List[Dict[str, Any]] = []
    box_counter = 0

    for packer in packers:
        for bin_obj in getattr(packer, "bins", []):
            items = list(getattr(bin_obj, "items", []))
            if not items:
                continue

            box_counter += 1
            # Group by Z layers
            layers = _cluster_layers_by_z(items, tol=layer_tol)

            # Steps start at 1 for each box
            step_num = 1
            for layer_items in layers:
                # Sort within layer for a consistent placement workflow:
                # front-to-back (y ascending), then left-to-right (x ascending) by default.
                if within_layer_sort == "xy":
                    layer_items.sort(key=lambda it: (_item_pos(it)[0], _item_pos(it)[1]))
                else:  # "yx"
                    layer_items.sort(key=lambda it: (_item_pos(it)[1], _item_pos(it)[0]))

                for it in layer_items:
                    x, y, z = _item_pos(it)
                    w, l, h = _item_dims_xyz(it)  # x→w, y→l, z→h
                    orient = "x-long" if w >= l else "y-long"
                    ident = getattr(it, "name", None) or getattr(it, "partno", "")

                    out.append(
                        {
                            "box": box_counter,
                            "step": step_num,
                            "id": ident,
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "w": float(w),
                            "l": float(l),
                            "h": float(h),
                            "orientation": orient,
                        }
                    )
                step_num += 1

    return out

def choose_container(
    padded_mailers: List[str],
    boxes: List[str],
    items: List[Dict[str, Any]],
    debug: bool = False,
    flat_height: float = 3,
    box_offset: int = 2,
    mailer_offset: int = 2,
    results: List[Dict[str, Any]] = [],
) -> Tuple[str, Optional[Any]]:
    """Select the smallest container that can hold all items.

    The search proceeds in two stages.  First, each padded mailer is
    considered in ascending order of area.  A simple 2D shelf
    algorithm is used to test whether all items can fit without
    stacking.  Only mailers whose dimensions and the items' smallest
    dimension satisfy the height constraint (four inches) are
    considered.  If a mailer is found, its dimension string is
    returned along with a list of placement descriptions when debug is
    enabled.  Otherwise, the algorithm falls back to 3D packing using
    the 3D bin packing library.  Each box candidate is tested in
    ascending order of volume until one succeeds.  On success the
    dimension string of the box is returned along with the packer
    instance when debug is enabled.

    Parameters
    ----------
    padded_mailers: List[str]
        A list of padded mailer sizes (``"WxL"``).  The height of
        every mailer is implicitly four inches and items may not be
        stacked.
    boxes: List[str]
        A list of box sizes (``"WxLxH"``).  Items may be rotated and
        stacked freely within boxes.
    items: List[Dict[str, Any]]
        A list of item descriptions.  Each dictionary must contain
        keys ``id``, ``dimension`` and ``quantity``.
    debug: bool, optional
        When ``True`` the function produces visualisations of the
        individual items and the chosen packing solution.

    Returns
    -------
    Tuple[str, Optional[Any]]
        A tuple containing the selected container dimension string and
        either a list of placement descriptions (for mailers) or the
        packer instance used (for boxes).  When debug is ``False`` the
        second element of the tuple is ``None``.

    Raises
    ------
    RuntimeError
        If no suitable container can be found.
    """
    # Expand the item list into individual instances.
    item_instances = generate_item_instances(items)
    # Try padded mailers first.
    for width, length, m_string in sort_mailers(padded_mailers):
        fits, placement = can_fit_in_mailer(
            item_instances, width - mailer_offset, length - mailer_offset, flat_height=flat_height,
        )
        if fits:
            if debug:
                # Visualise each item and the solution.
                # visualise_individual_items(item_instances)
                if placement is not None:
                    visualise_mailer_solution(
                        (width, length), placement, mailer_offset, flat_height
                    )
                    
            results.append({
                "box_dimension": f"{width}x{length}",
                "packers": mailer_arrangement_json(placement),
                "unfit_items": []
            })
            return results
        
    # Fallback to boxes.
    for w, h, d, b_string in sort_boxes(boxes):
        # Try all permutations of the box dimensions.  The 3D bin
        # packing algorithm treats the dimensions as fixed, so by
        # permuting them we emulate rotating the box itself.  This
        # matters when the list of item orientations is such that
        # different container orientations yield different packing
        # success.
        for perm in set(
            itertools.permutations((w - box_offset, h - box_offset, d - box_offset))
        ):
            success, packer = pack_in_box(item_instances, perm)
            if success:
                if debug and packer is not None:
                    # visualise_individual_items(item_instances)
                    visualise_box_solution(packer, perm, box_offset)
                # Return the canonical dimension string in sorted order
                sorted_dims = sorted(perm)
                results.append({
                    "box_dimension": f"{sorted_dims[0] + box_offset}x{sorted_dims[1] + box_offset}x{sorted_dims[2] + box_offset}",
                    "packers": boxes_arrangement_json([packer], layer_tol) if debug else None,
                    "unfit_items": []
                })
                return results
            
    # ---- Greedy multi-box fallback with the largest box ----
    multi_str, multi_packers, unfitItems, last_box_items = greedy_pack_into_biggest_box(
        item_instances, boxes, debug=debug, box_offset=box_offset
    )
    if last_box_items:
        choose_container(
            padded_mailers=padded_mailers,
            boxes=boxes,
            items=last_box_items,
            debug=debug,
            flat_height=flat_height,
            box_offset=box_offset,
            mailer_offset=mailer_offset,
            results=results
        )

    results.append({
        "box_dimension": multi_str,
        "packers": boxes_arrangement_json(multi_packers, layer_tol),
        "unfit_items": unfitItems
    })

    # return multi_str, multi_packers, unfitItems
    return results


def _dim_to_str(dims: Tuple[float, float, float]) -> str:
    """Format dims back to 'WxHxD' with ints when values are integral."""
    def fmt(x: float) -> str:
        xi = int(round(x))
        return str(xi) if abs(x - xi) < 1e-6 else f"{x:g}"
    return "x".join(fmt(v) for v in dims)

def summarize_item_instances(instances: List["ItemInstance"]) -> List[Dict[str, Any]]:
    """
    Aggregate ItemInstance objects into [{'id','dimension','quantity'}, ...].
    Groups by (id, dimension).
    """
    counts: defaultdict[Tuple[str, str], int] = defaultdict(int)
    for inst in instances:
        item_id = getattr(inst, "id", None) or getattr(inst, "name", "UNKNOWN")
        # Prefer any stored string dimension if you have one; else build from dims tuple
        dim_str = (
            getattr(inst, "dimension", None)
            or getattr(inst, "dim_str", None)
            or _dim_to_str(tuple(getattr(inst, "dims")))  # expects inst.dims = (W,H,D)
        )
        counts[(item_id, dim_str)] += 1

    # Build the desired list of dicts
    out: List[Dict[str, Any]] = [
        {"id": item_id, "dimension": dim_str, "quantity": qty}
        for (item_id, dim_str), qty in counts.items()
    ]
    # Optional: stable ordering
    out.sort(key=lambda d: (d["id"], d["dimension"]))
    return out

def iter_bins(packer):
    """Return a list of bins from a py3dbp Packer across versions."""
    return list(packer) if hasattr(packer, "__iter__") else getattr(packer, "bins", [])


if __name__ == "__main__":
    # Entry point for the script
    results = choose_container(
        padded_mailers=["6x9", "10x13"],
        boxes=["8x8x8", "12x10x6", "14x12x10"],
        items=[
            {"id": "A", "dimension": "6x4x2", "quantity": 8},
            {"id": "B", "dimension": "10x3x2", "quantity": 10},
            {"id": "C", "dimension": "15x15x12", "quantity": 3},
        ],
        debug=False,
        box_offset=2,
    )
    pprint(results)