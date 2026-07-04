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


@dataclass(frozen=True)
class ChannelStack:
    """A microscopy image kept at full channel depth.

    ``data`` is ``H×W×C`` float32 (channel-last, never normalised). ``names``
    labels each channel for the UI/measurements; it always has length ``C``.
    """
    data: np.ndarray
    names: list[str]

    @property
    def n_channels(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]

    def channel(self, idx: int) -> np.ndarray:
        return self.data[:, :, idx]


def _default_names(n: int) -> list[str]:
    return [f"Channel {i}" for i in range(n)]


def to_channel_stack(array: np.ndarray,
                     channel_axis: int | None = None,
                     names: Sequence[str] | None = None) -> ChannelStack:
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
    return ChannelStack(data=data, names=names)


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
    from data.utils import read_file_to_numpy
    return to_channel_stack(read_file_to_numpy(path))


def _read_tiff_stack(path: Path) -> ChannelStack:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        arr = series.asarray()
        axes = series.axes or ""
        names = _ome_channel_names(tf)

    if not axes or len(axes) != arr.ndim:
        return to_channel_stack(arr, names=None)

    arr, axes = _reduce_extra_axes(arr, axes)
    if "C" in axes:
        cax = axes.index("C")
    elif arr.ndim == 3:
        # No labelled channel axis but a 3-D plane → fall back to the heuristic.
        return to_channel_stack(arr)
    else:
        cax = None

    stack = to_channel_stack(arr, channel_axis=cax)
    if names is not None and len(names) == stack.n_channels:
        stack = ChannelStack(data=stack.data, names=list(names))
    return stack


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
