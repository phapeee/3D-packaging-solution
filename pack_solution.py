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
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    from py3dbp import Packer, Bin, Item, Painter  # type: ignore
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
    dims: Tuple[int, int, int]


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
        dims_raw = parse_dims(itm["dimension"])
        if len(dims_raw) != 3:
            raise ValueError(f"Items must have three dimensions, got {itm}")
        # store dimensions in ascending order for easier orientation
        dims_sorted = tuple(sorted(dims_raw))  # type: ignore
        for _ in range(itm.get("quantity", 1)):
            instances.append(ItemInstance(id=itm["id"], dims=dims_sorted))
    return instances


def can_fit_in_mailer(
    items: List[ItemInstance], mailer_width: int, mailer_length: int
) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    """Attempt to pack items into a single padded mailer.

    A padded mailer is treated as a 2D rectangle with fixed height
    (four inches).  Items may not be stacked, therefore their
    smallest dimension (thickness) must not exceed four.  The helper
    uses a simple shelf algorithm to place rectangles on the x‑y plane.
    Each item can be rotated in the plane, so the two largest
    dimensions of each item become the potential width/length.  The
    mailer is represented with the smaller dimension as the width and
    the larger as the length.  Items are sorted by descending maximum
    side before placement to improve packing efficiency.

    Parameters
    ----------
    items: List[ItemInstance]
        List of individual items to be placed.
    mailer_width: int
        Width of the mailer (smaller of the two supplied dimensions).
    mailer_length: int
        Length of the mailer (larger of the two supplied dimensions).

    Returns
    -------
    Tuple[bool, Optional[List[Dict[str, Any]]]]
        A boolean indicating whether all items fit, and if so a list
        describing the placement of each item.  Each placement
        dictionary contains the original :class:`ItemInstance`, the
        orientation used (0 means largest dimension along x, 1 means
        along y) and the ``x``/``y`` position of the lower left
        corner within the mailer.  The placement list is ordered
        according to the sort order used during placement, not
        necessarily the original input order.
    """
    # Filter out items that are too thick for the mailer.
    for inst in items:
        if inst.dims[0] > 4:
            return False, None
    # Sort items by descending area (largest face) then by max side.
    # dims are sorted ascending, so dims[2]*dims[1] is the area of the
    # two largest sides.
    items_sorted = sorted(
        items, key=lambda it: (it.dims[2] * it.dims[1], it.dims[2]), reverse=True
    )
    placements: List[Dict[str, Any]] = []
    y_offset = 0
    row_height = 0
    row_width = 0
    for inst in items_sorted:
        placed = False
        # orientation 0: width=dims[2], length=dims[1]; orientation 1: width=dims[1], length=dims[2]
        orientations = [
            (inst.dims[2], inst.dims[1], 0),
            (inst.dims[1], inst.dims[2], 1),
        ]
        for width_i, length_i, orientation_flag in orientations:
            # try to place in current row
            if (
                row_width + width_i <= mailer_width
                and length_i <= mailer_length - y_offset
            ):
                placements.append(
                    {
                        "item": inst,
                        "orientation": orientation_flag,
                        "position": (row_width, y_offset),
                        "width": width_i,
                        "length": length_i,
                    }
                )
                row_width += width_i
                row_height = max(row_height, length_i)
                placed = True
                break
        if not placed:
            # start a new row
            y_offset += row_height
            # reset row
            row_width = 0
            row_height = 0
            # Check if there is enough remaining length for the item.
            # Try orientations again.
            for width_i, length_i, orientation_flag in orientations:
                if width_i <= mailer_width and length_i <= mailer_length - y_offset:
                    placements.append(
                        {
                            "item": inst,
                            "orientation": orientation_flag,
                            "position": (row_width, y_offset),
                            "width": width_i,
                            "length": length_i,
                        }
                    )
                    row_width += width_i
                    row_height = max(row_height, length_i)
                    placed = True
                    break
            if not placed:
                return False, None
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
            number_of_decimals=0,
        )
        success = len(packer.unfit_items) == 0
    except Exception:
        success = False
        packer = None
    return success, packer if success else None


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
        draw_cuboid(ax, (x, y, 0), (dx, dy, dz), color=color, label=inst.id)

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
    """Convert internal placement records into a JSON-serializable list."""
    out = []
    for p in placements or []:
        inst = p["item"]
        out.append(
            {
                "id": inst.id,  # original item id
                "x": p["position"][0],  # lower-left corner within mailer (in)
                "y": p["position"][1],
                "w": p["width"],  # placed width (in)
                "l": p["length"],  # placed length (in)
                "orientation": "x-long" if p["orientation"] == 0 else "y-long",
            }
        )
    return out


def choose_container(
    padded_mailers: List[str],
    boxes: List[str],
    items: List[Dict[str, Any]],
    debug: bool = False,
    flat_height: float = 3,
    box_offset: int = 2,
    mailer_offset: int = 2,
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
            item_instances, width - mailer_offset, length - mailer_offset
        )
        if fits:
            if debug:
                # Visualise each item and the solution.
                # visualise_individual_items(item_instances)
                if placement is not None:
                    visualise_mailer_solution(
                        (width, length), placement, mailer_offset, flat_height
                    )
            return f"{width}x{length}", mailer_arrangement_json(placement)
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
                return (
                    f"{sorted_dims[0] + box_offset}x{sorted_dims[1] + box_offset}x{sorted_dims[2] + box_offset}",
                    packer if debug else None,
                )
    raise RuntimeError("No suitable mailer or box could be found to fit all items")


def iter_bins(packer):
    """Return a list of bins from a py3dbp Packer across versions."""
    return list(packer) if hasattr(packer, "__iter__") else getattr(packer, "bins", [])
