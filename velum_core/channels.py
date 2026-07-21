"""
Multi-channel microscopy support.

Real microscopy is N-channel (DAPI + membrane + one or more marker channels),
but the CellSeg1 read path (``data.utils.read_image_to_numpy`` /
``cv2.imread``) collapses everything to an 8-bit RGB frame and throws the extra
channels — and their quantitative intensity — away. This module keeps the raw
channel stack around so the user can

  * open an OME-TIFF / multi-page TIFF with more than three channels,
  * pick which channel(s) drive segmentation,
  * normalise each channel independently by percentiles, and
  * measure per-channel intensity under every segmented cell.

Everything here is pure and engine-agnostic: it maps files/arrays to a
canonical ``H×W×C`` float stack and projects a chosen subset to the ``H×W×3``
uint8 frame the segmenter already expects, so the existing RGB path is never
touched. The napari widget opts in only when a stack actually has channels to
choose from; ordinary RGB / grayscale images keep flowing through the old path
byte-for-byte.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np


class MissingReaderError(RuntimeError):
    """A microscopy format needs an optional reader that isn't installed.

    Raised (instead of crashing with an ``ImportError``) so the UI can show a
    single actionable line — "install ``nd2`` to open .nd2 files" — and unknown
    formats degrade gracefully rather than taking the app down.
    """


@dataclass(frozen=True)
class ChannelStack:
    """A microscopy image kept at full channel depth.

    ``data`` is ``H×W×C`` float32 (channel-last, never normalised). ``names``
    labels each channel for the UI/measurements; it always has length ``C``.
    ``pixel_size_um`` is the physical pixel size in microns read from the file's
    metadata (``None`` when unknown), so measurements can auto-fill µm/pixel.
    """
    data: np.ndarray
    names: list[str]
    pixel_size_um: float | None = None

    @property
    def n_channels(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]

    def channel(self, idx: int) -> np.ndarray:
        return self.data[:, :, idx]


@dataclass(frozen=True)
class VolumeStack:
    """A z-stack or time-lapse kept as one :class:`ChannelStack`-shaped frame
    per plane, instead of collapsing the stack axis to its first plane the
    way :func:`read_channel_stack` does.

    ``data`` is ``Z×H×W×C`` float32 (channel-last, never normalised); ``names``
    labels each channel (shared across every plane — channel identity doesn't
    vary slice to slice). Used by the z-stack / time-lapse prediction path
    (:mod:`velum_core.volume_stitch` stitches the resulting per-plane
    instance masks back into one consistent volume).
    """
    data: np.ndarray
    names: list[str]
    pixel_size_um: float | None = None

    @property
    def n_planes(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[3]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]

    def plane(self, z: int) -> ChannelStack:
        return ChannelStack(data=self.data[z], names=self.names,
                            pixel_size_um=self.pixel_size_um)


def _default_names(n: int) -> list[str]:
    return [f"Channel {i}" for i in range(n)]


def to_channel_stack(array: np.ndarray,
                     channel_axis: int | None = None,
                     names: Sequence[str] | None = None,
                     pixel_size_um: float | None = None) -> ChannelStack:
    """Canonicalise an arbitrary image array to a channel-last ``ChannelStack``.

    ``channel_axis`` pins which axis holds channels; when ``None`` it is
    inferred as the smallest axis of a 3-D array *if* that axis is small
    (``<= 8``) and strictly smaller than both other axes — the usual
    ``C×H×W`` / ``H×W×C`` microscopy layouts. A 2-D array becomes a single
    channel. Leading singleton axes (a ``1×C×H×W`` OME page) are squeezed;
    genuinely higher-dimensional stacks (z / t) must be reduced by the caller
    or pinned via ``channel_axis`` — otherwise a ``ValueError`` is raised so we
    never silently segment the wrong plane.
    """
    arr = np.asarray(array)
    # Drop only leading singleton axes so an explicit channel_axis stays valid.
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = np.squeeze(arr) if arr.ndim > 3 else arr

    if arr.ndim == 2:
        data = arr[:, :, None]
    elif arr.ndim == 3:
        axis = channel_axis
        if axis is None:
            small = int(np.argmin(arr.shape))
            others = [arr.shape[i] for i in range(3) if i != small]
            axis = small if (arr.shape[small] <= 8
                             and arr.shape[small] < min(others)) else 2
        axis = axis % 3
        data = np.moveaxis(arr, axis, 2)
    else:
        raise ValueError(
            f"Cannot infer channels for shape {arr.shape}; pass channel_axis "
            "or reduce z/t dimensions first.")

    data = np.ascontiguousarray(data, dtype=np.float32)
    c = data.shape[2]
    if names is not None:
        names = list(names)
        if len(names) != c:
            raise ValueError(f"names has {len(names)} entries for {c} channels")
    else:
        names = _default_names(c)
    return ChannelStack(data=data, names=names, pixel_size_um=pixel_size_um)


def stack_from_axes_array(arr: np.ndarray,
                          axes: str,
                          names: Sequence[str] | None = None,
                          pixel_size_um: float | None = None) -> ChannelStack:
    """Build a :class:`ChannelStack` from an array plus its tifffile-style axis
    string (e.g. ``"ZCYX"``).

    Non-spatial, non-channel axes (Z/T/S/P/...) are reduced to their first
    plane; the ``C`` axis (when present) becomes the channel axis, otherwise a
    3-D plane falls back to the shape heuristic. Shared by every reader
    (TIFF/OME/ND2/CZI/LIF) so channel-axis handling lives in one place.
    """
    if not axes or len(axes) != arr.ndim:
        return to_channel_stack(arr, names=names, pixel_size_um=pixel_size_um)

    arr, axes = _reduce_extra_axes(arr, axes)
    if "C" in axes:
        cax = axes.index("C")
    elif arr.ndim == 3:
        # No labelled channel axis but a 3-D plane → fall back to the heuristic.
        return to_channel_stack(arr, names=names, pixel_size_um=pixel_size_um)
    else:
        cax = None

    stack = to_channel_stack(arr, channel_axis=cax, pixel_size_um=pixel_size_um)
    if names is not None and len(names) == stack.n_channels:
        stack = ChannelStack(data=stack.data, names=list(names),
                             pixel_size_um=pixel_size_um)
    return stack


def stack_from_axes_array_zstack(arr: np.ndarray,
                                 axes: str,
                                 names: Sequence[str] | None = None,
                                 pixel_size_um: float | None = None) -> VolumeStack:
    """Like :func:`stack_from_axes_array` but keeps the Z (or T) axis as a
    leading stack dimension instead of reducing it to its first plane.

    Picks ``Z`` over ``T`` when a file has both (a rare 5-D ZTCYX acquisition
    reduces its T axis to the first timepoint, keeping Z — segmentation cares
    about the stack it was asked to segment, not every axis a file has). Each
    resulting plane is still run through :func:`stack_from_axes_array`, so a
    z-stack *and* multi-channel image (e.g. ``"ZCYX"``) keeps per-plane
    channel handling identical to the single-plane path. A file with no Z/T
    axis at all (or no axis metadata) degrades to a single-plane
    :class:`VolumeStack` so callers never need a separate code path.
    """
    if not axes or len(axes) != arr.ndim:
        stack = to_channel_stack(arr, names=names, pixel_size_um=pixel_size_um)
        return VolumeStack(data=stack.data[None], names=stack.names,
                           pixel_size_um=pixel_size_um)

    stack_axis = "Z" if "Z" in axes else ("T" if "T" in axes else None)
    if stack_axis is None:
        stack = stack_from_axes_array(arr, axes, names=names, pixel_size_um=pixel_size_um)
        return VolumeStack(data=stack.data[None], names=stack.names,
                           pixel_size_um=pixel_size_um)

    zax = axes.index(stack_axis)
    arr = np.moveaxis(arr, zax, 0)
    plane_axes = axes[:zax] + axes[zax + 1:]

    planes = []
    plane_names = names
    for i in range(arr.shape[0]):
        plane_stack = stack_from_axes_array(arr[i], plane_axes, names=plane_names,
                                            pixel_size_um=pixel_size_um)
        plane_names = plane_stack.names   # keep names consistent across planes
        planes.append(plane_stack.data)
    return VolumeStack(data=np.stack(planes, axis=0), names=plane_names,
                       pixel_size_um=pixel_size_um)


def _reduce_extra_axes(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Collapse non-spatial, non-channel axes (Z, T, S, ...) by taking index 0.

    ``axes`` is a tifffile axis string aligned to ``arr`` (e.g. ``"ZCYX"``).
    We keep the first plane rather than max-projecting so behaviour is
    deterministic and a caller who wants a projection can do it explicitly.
    """
    keep = {"Y", "X", "C"}
    while len(axes) > 3:
        drop = next((i for i, a in enumerate(axes) if a not in keep), None)
        if drop is None:
            break
        arr = np.take(arr, 0, axis=drop)
        axes = axes[:drop] + axes[drop + 1:]
    return arr, axes


def read_channel_stack(path: Union[str, Path]) -> ChannelStack:
    """Read an image file into a full-depth :class:`ChannelStack`.

    TIFF / OME-TIFF are read with ``tifffile``, using the series' axis string
    to locate the channel axis and read channel names from OME metadata when
    present; z/t/sample axes are reduced to their first plane. Other formats
    fall back to a plain read plus the :func:`to_channel_stack` heuristic.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return _read_tiff_stack(path)
    if suffix == ".nd2":
        return _read_nd2_stack(path)
    if suffix == ".czi":
        return _read_czi_stack(path)
    if suffix == ".lif":
        return _read_lif_stack(path)
    from data.utils import read_file_to_numpy
    return to_channel_stack(read_file_to_numpy(path))


def _read_tiff_stack(path: Path) -> ChannelStack:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        arr = series.asarray()
        axes = series.axes or ""
        names = _ome_channel_names(tf)
        pixel_size = _tiff_pixel_size_um(tf)

    return stack_from_axes_array(arr, axes, names=names, pixel_size_um=pixel_size)


def read_volume_stack(path: Union[str, Path]) -> VolumeStack:
    """Read a z-stack or time-lapse file, keeping every plane.

    TIFF/OME-TIFF, ND2, and CZI all keep their real Z/T axis structure (via
    :func:`stack_from_axes_array_zstack`, shared with the channel-only path
    so per-plane channel handling is identical either way). LIF does too when
    its per-plane dimensions can be read (:func:`_read_lif_volume`); any
    format with no Z/T axis at all, or one this function doesn't specifically
    handle, degrades to a 1-plane :class:`VolumeStack` via
    :func:`read_channel_stack` so the caller never needs a separate branch.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            arr = series.asarray()
            axes = series.axes or ""
            names = _ome_channel_names(tf)
            pixel_size = _tiff_pixel_size_um(tf)
        return stack_from_axes_array_zstack(arr, axes, names=names, pixel_size_um=pixel_size)
    if suffix == ".nd2":
        return _read_nd2_volume(path)
    if suffix == ".czi":
        return _read_czi_volume(path)
    if suffix == ".lif":
        return _read_lif_volume(path)
    stack = read_channel_stack(path)
    return VolumeStack(data=stack.data[None], names=stack.names,
                       pixel_size_um=stack.pixel_size_um)


def _require(module: str, fmt: str, package: str | None = None):
    """Import an optional reader or raise a friendly :class:`MissingReaderError`.

    ``package`` names the pip distribution when it differs from the import name,
    so the message tells the user exactly what to install to open ``fmt``.
    """
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - trivial, exercised via mocks
        pkg = package or module
        raise MissingReaderError(
            f"Reading {fmt} files needs the optional '{pkg}' package. "
            f"Install it with:  pip install {pkg}") from exc


def _nd2_raw(path: Path):
    """Open a ``.nd2`` file and return ``(arr, axes, names, pixel_size_um)``,
    shared by the channel-only and volume-keeping read paths."""
    nd2 = _require("nd2", ".nd2")
    with nd2.ND2File(str(path)) as f:
        arr = np.asarray(f.asarray())
        axes = "".join(getattr(f, "sizes", {}).keys())
        names = _nd2_channel_names(f)
        pixel_size = _nd2_pixel_size_um(f)
    return arr, axes, names, pixel_size


def _read_nd2_stack(path: Path) -> ChannelStack:
    """Read a Nikon ``.nd2`` file via the optional ``nd2`` package."""
    arr, axes, names, pixel_size = _nd2_raw(path)
    return stack_from_axes_array(arr, axes, names=names, pixel_size_um=pixel_size)


def _read_nd2_volume(path: Path) -> VolumeStack:
    """Read a Nikon ``.nd2`` z-stack/time-lapse, keeping every plane."""
    arr, axes, names, pixel_size = _nd2_raw(path)
    return stack_from_axes_array_zstack(arr, axes, names=names, pixel_size_um=pixel_size)


def _czi_raw(path: Path):
    """Open a ``.czi`` file and return ``(arr, axes, pixel_size_um)``, shared
    by the channel-only and volume-keeping read paths."""
    czifile = _require("czifile", ".czi")
    with czifile.CziFile(str(path)) as f:
        arr = np.asarray(f.asarray())
        axes = f.axes or ""
        pixel_size = _czi_pixel_size_um(f)
    # czifile pads with singleton acquisition axes (B/V/M/0/...); drop every
    # length-1 axis that isn't a real Y/X/C plane so the axis string aligns.
    # A genuine multi-plane Z/T axis (size > 1) is never dropped here, so the
    # volume path below still sees it.
    drop = tuple(i for i, s in enumerate(arr.shape)
                 if s == 1 and (i >= len(axes) or axes[i] not in "YXC"))
    arr = np.squeeze(arr, axis=drop)
    axes = "".join(axes[i] for i in range(len(axes)) if i not in drop)
    return arr, axes, pixel_size


def _read_czi_stack(path: Path) -> ChannelStack:
    """Read a Zeiss ``.czi`` file via the optional ``czifile`` package."""
    arr, axes, pixel_size = _czi_raw(path)
    return stack_from_axes_array(arr, axes, pixel_size_um=pixel_size)


def _read_czi_volume(path: Path) -> VolumeStack:
    """Read a Zeiss ``.czi`` z-stack/time-lapse, keeping every plane."""
    arr, axes, pixel_size = _czi_raw(path)
    return stack_from_axes_array_zstack(arr, axes, pixel_size_um=pixel_size)


def _read_lif_stack(path: Path) -> ChannelStack:
    """Read a Leica ``.lif`` file (first image) via the optional ``readlif``."""
    readlif = _require("readlif.reader", ".lif", package="readlif")
    lif = readlif.LifFile(str(path))
    img = lif.get_image(0)
    frames = [np.asarray(p) for p in img.get_iter_c()]  # one plane per channel
    arr = np.stack(frames, axis=0)                      # C,H,W
    pixel_size = _lif_pixel_size_um(img)
    return stack_from_axes_array(arr, "CYX", pixel_size_um=pixel_size)


def _read_lif_volume(path: Path) -> VolumeStack:
    """Read a Leica ``.lif`` z-stack (first image), keeping every Z plane.

    readlif's per-plane API (``LifImage.dims.z`` / ``.get_frame(z=, c=)``)
    isn't exercised against a real ``.lif`` file anywhere in this codebase's
    test data, so this can't be verified against the real package here —
    if the z-plane count can't be read, or is 1 (no real z-stack), this
    degrades to exactly :func:`_read_lif_stack`'s existing channel-only
    behaviour wrapped as a 1-plane volume, rather than risking a wrong
    attribute-name guess crashing the read entirely.
    """
    readlif = _require("readlif.reader", ".lif", package="readlif")
    lif = readlif.LifFile(str(path))
    img = lif.get_image(0)
    pixel_size = _lif_pixel_size_um(img)

    try:
        nz = int(img.dims.z)
        nc = int(img.channels)
        if nz <= 1:
            raise ValueError("no multi-plane z-stack in this .lif image")
        planes = []
        for z in range(nz):
            frames = [np.asarray(img.get_frame(z=z, c=c)) for c in range(nc)]
            planes.append(np.stack(frames, axis=0))            # C,H,W at this z
        arr = np.stack(planes, axis=0)                          # Z,C,H,W
        return stack_from_axes_array_zstack(arr, "ZCYX", pixel_size_um=pixel_size)
    except Exception:
        frames = [np.asarray(p) for p in img.get_iter_c()]
        arr = np.stack(frames, axis=0)
        stack = stack_from_axes_array(arr, "CYX", pixel_size_um=pixel_size)
        return VolumeStack(data=stack.data[None], names=stack.names,
                           pixel_size_um=pixel_size)


def _ome_channel_names(tf) -> list[str] | None:
    """Best-effort channel names from OME-XML; ``None`` when unavailable."""
    meta = getattr(tf, "ome_metadata", None)
    if not meta:
        return None
    try:
        import re
        names = re.findall(r'<Channel[^>]*\bName="([^"]*)"', meta)
        return names or None
    except Exception:
        return None


# ── physical pixel size (µm/pixel) ───────────────────────────────────────────

# Conversion factors to microns for the unit strings microscopy metadata uses.
_UM_PER_UNIT = {
    "m": 1e6, "meter": 1e6, "metre": 1e6,
    "cm": 1e4, "centimeter": 1e4,
    "mm": 1e3, "millimeter": 1e3,
    "µm": 1.0, "um": 1.0, "micron": 1.0, "micrometer": 1.0, "micrometre": 1.0,
    "nm": 1e-3, "nanometer": 1e-3,
    "å": 1e-4, "angstrom": 1e-4, "a": 1e-4,
    "in": 25400.0, "inch": 25400.0,
}


def physical_size_to_um(value: float, unit: str | None) -> float | None:
    """Convert a physical pixel size given in ``unit`` to microns.

    Returns ``None`` for a non-positive size or an unrecognised unit rather than
    guessing, so a bad/absent unit never silently poisons measurements. A
    missing unit defaults to microns (the OME-TIFF default is ``µm``).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if not unit:
        return value  # OME default unit is micron
    factor = _UM_PER_UNIT.get(unit.strip().lower().rstrip("s"))
    if factor is None:
        return None
    return value * factor


def read_pixel_size_um(path: Union[str, Path]) -> float | None:
    """Best-effort physical pixel size (µm/pixel) from a file's metadata.

    Reads OME-XML ``PhysicalSizeX`` / TIFF resolution tags for TIFFs and the
    native voxel metadata for ND2/CZI/LIF. Returns ``None`` when the file
    carries no calibration (e.g. a plain PNG) so the UI leaves the field at 0.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(path) as tf:
                return _tiff_pixel_size_um(tf)
        if suffix in (".nd2", ".czi", ".lif"):
            return read_channel_stack(path).pixel_size_um
    except Exception:
        return None
    return None


def _tiff_pixel_size_um(tf) -> float | None:
    """Pixel size from OME-XML first, then baseline TIFF resolution tags."""
    ome = _ome_pixel_size_um(getattr(tf, "ome_metadata", None))
    if ome is not None:
        return ome
    try:
        page = tf.pages[0]
        tags = page.tags
    except Exception:
        return None
    res = tags.get("XResolution")
    if res is None or not res.value:
        return None
    num, den = res.value if isinstance(res.value, tuple) else (res.value, 1)
    if not num:
        return None
    pixels_per_unit = num / den                       # e.g. pixels per cm
    unit_tag = tags.get("ResolutionUnit")
    unit_code = getattr(unit_tag, "value", 2)
    unit_code = int(getattr(unit_code, "value", unit_code))
    unit = {2: "inch", 3: "cm"}.get(unit_code)
    if unit is None:                                  # 1 = no absolute unit
        return None
    unit_um = _UM_PER_UNIT[unit]
    return unit_um / pixels_per_unit                  # µm per pixel


def _ome_pixel_size_um(meta: str | None) -> float | None:
    """Parse ``PhysicalSizeX`` (+ its unit) out of an OME-XML string."""
    if not meta:
        return None
    import re
    m = re.search(r'PhysicalSizeX="([^"]+)"', meta)
    if not m:
        return None
    unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', meta)
    return physical_size_to_um(m.group(1), unit.group(1) if unit else None)


def _nd2_channel_names(f) -> list[str] | None:
    try:
        names = [c.channel.name for c in f.metadata.channels]
        return [n for n in names if n] or None
    except Exception:
        return None


def _nd2_pixel_size_um(f) -> float | None:
    try:
        vs = f.voxel_size()          # named tuple in microns (x, y, z)
        return physical_size_to_um(vs.x, "um")
    except Exception:
        return None


def _czi_pixel_size_um(f) -> float | None:
    """CZI stores scaling in metres under Scaling/Items/Distance[Id='X']."""
    try:
        import re
        meta = f.metadata() if callable(getattr(f, "metadata", None)) else f.metadata
        m = re.search(
            r'<Distance[^>]*Id="X"[^>]*>.*?<Value>([^<]+)</Value>', meta, re.S)
        if not m:
            return None
        return physical_size_to_um(float(m.group(1)) * 1e6, "um")  # m → µm
    except Exception:
        return None


def _lif_pixel_size_um(img) -> float | None:
    try:
        px_per_um = float(img.scale[0])   # readlif: pixels per micron on X
        if px_per_um > 0:
            return 1.0 / px_per_um
    except Exception:
        pass
    return None


def probe_channels(path: Union[str, Path]) -> tuple[int, list[str]]:
    """Cheaply report ``(n_channels, names)`` without loading pixel data.

    For TIFF/OME-TIFF this reads only the series shape/axes and OME channel
    names; for other formats it falls back to a full read. Returns ``(1, ...)``
    for a plain grayscale image and ``(3, ...)`` for ordinary RGB, so the UI
    can decide whether a channel picker is worth showing (i.e. a genuine
    multi-channel stack, not a normal photo).
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tf:
                series = tf.series[0]
                axes, shape = series.axes or "", series.shape
                names = _ome_channel_names(tf)
            if axes and "C" in axes and len(axes) == len(shape):
                c = int(shape[axes.index("C")])
            elif len(shape) == 3:
                c = int(min(shape)) if min(shape) <= 8 else int(shape[-1])
            else:
                c = 1
            if names and len(names) == c:
                return c, list(names)
            return c, _default_names(c)
        except Exception:
            pass
    stack = read_channel_stack(path)
    return stack.n_channels, stack.names


def _axes_has_multiplane(axes: str, shape: tuple) -> bool:
    """Whether an axes string + shape pair has a genuine (length > 1) Z or T
    axis. Shared by every format's has_z_stack branch below."""
    if not axes or len(axes) != len(shape):
        return False
    return any(ax in axes and shape[axes.index(ax)] > 1 for ax in ("Z", "T"))


def has_z_stack(path: Union[str, Path]) -> bool:
    """Cheaply report whether a file has a genuine multi-plane Z or T axis.

    TIFF/OME-TIFF, ND2, and CZI read only shape/axis metadata (no pixel
    data), mirroring :func:`probe_channels`'s "don't load the full array"
    shape, so the UI can decide whether a "segment as z-stack" toggle is even
    worth offering. LIF reads its lightweight per-image dimension descriptor
    (see :func:`_read_lif_volume` for why this can't be verified against a
    real file here). ``False`` for any other format, any file with no axis
    metadata, a Z/T axis of length 1 (a single-plane file with incidental Z/T
    metadata shouldn't offer a no-op toggle), a missing optional reader
    package, or any read/parse failure — this must never raise, only ever
    suggest the toggle when it's actually useful.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(path) as tf:
                series = tf.series[0]
                axes, shape = series.axes or "", series.shape
            return _axes_has_multiplane(axes, shape)
        if suffix == ".nd2":
            nd2 = _require("nd2", ".nd2")
            with nd2.ND2File(str(path)) as f:
                sizes = dict(getattr(f, "sizes", {}))
            axes = "".join(sizes.keys())
            shape = tuple(sizes.values())
            return _axes_has_multiplane(axes, shape)
        if suffix == ".czi":
            czifile = _require("czifile", ".czi")
            with czifile.CziFile(str(path)) as f:
                axes, shape = f.axes or "", f.shape
            return _axes_has_multiplane(axes, shape)
        if suffix == ".lif":
            readlif = _require("readlif.reader", ".lif", package="readlif")
            lif = readlif.LifFile(str(path))
            img = lif.get_image(0)
            return int(img.dims.z) > 1
        return False
    except Exception:
        return False


def percentile_normalize(channel: np.ndarray,
                         low: float = 1.0,
                         high: float = 99.0) -> np.ndarray:
    """Contrast-stretch one channel to ``uint8`` by percentile clipping.

    Maps the ``low``..``high`` intensity percentiles to 0..255. A flat channel
    (or one where the two percentiles coincide) returns all zeros rather than
    dividing by zero, so a dark/empty marker channel is handled gracefully.
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("require 0 <= low < high <= 100")
    chan = np.asarray(channel, dtype=np.float32)
    lo, hi = np.percentile(chan, [low, high])
    if hi <= lo:
        return np.zeros(chan.shape, dtype=np.uint8)
    scaled = (chan - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def project_to_rgb(stack: ChannelStack,
                   channels: Sequence[int] | None = None,
                   low: float = 1.0,
                   high: float = 99.0) -> np.ndarray:
    """Project selected channels to the ``H×W×3`` uint8 frame the engine wants.

    Each selected channel is percentile-normalised independently, then placed
    into an RGB slot:

      * 1 channel  → replicated to gray (matches the legacy single-channel path
        and gives SAM/Cellpose the achromatic input they expect),
      * 2 channels → red + green,
      * 3+ channels → first three into R, G, B (extra selections ignored).

    ``channels`` defaults to every channel (capped at the first three).
    """
    if channels is None:
        channels = list(range(min(stack.n_channels, 3)))
    else:
        channels = list(channels)
    if not channels:
        raise ValueError("select at least one channel")
    for c in channels:
        if not 0 <= c < stack.n_channels:
            raise ValueError(
                f"channel {c} out of range 0..{stack.n_channels - 1}")

    h, w = stack.shape
    if len(channels) == 1:
        gray = percentile_normalize(stack.channel(channels[0]), low, high)
        return np.stack([gray] * 3, axis=-1)

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for slot, c in enumerate(channels[:3]):
        rgb[:, :, slot] = percentile_normalize(stack.channel(c), low, high)
    return rgb


def channel_means(mask: np.ndarray,
                  stack: ChannelStack,
                  channels: Sequence[int] | None = None) -> dict[int, np.ndarray]:
    """Mean raw intensity of each label under each selected channel.

    Returns ``{channel_index: array indexed by label}`` where element ``k`` is
    the mean intensity of cell ``k`` (element 0, the background, is included so
    callers can index by label directly). Uses raw stack values, not the
    normalised projection, so the numbers are physically meaningful.
    """
    mask = np.asarray(mask)
    if mask.shape != stack.shape:
        raise ValueError(
            f"mask {mask.shape} does not match stack {stack.shape}")
    if channels is None:
        channels = list(range(stack.n_channels))
    labels = mask.ravel().astype(np.int64)
    n = int(labels.max()) + 1 if labels.size else 1
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid 0/0; empty labels report 0 mean
    out: dict[int, np.ndarray] = {}
    for c in channels:
        sums = np.bincount(labels, weights=stack.channel(c).ravel().astype(np.float64),
                           minlength=n)
        out[int(c)] = sums / counts
    return out
