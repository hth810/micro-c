"""Shared input, annotation, and contact-window utilities."""
from __future__ import annotations

import gzip
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


TYPES = ("CHIN", "OPCID", "CHID")


@contextmanager
def open_cool(path: str | Path):
    """Open .cool or .cool.gz without leaving an extracted copy behind."""
    import cooler

    path = Path(path)
    if path.suffix != ".gz":
        yield cooler.Cooler(str(path))
        return
    with tempfile.TemporaryDirectory(prefix="microc_cool_") as folder:
        target = Path(folder) / path.with_suffix("").name
        with gzip.open(path, "rb") as source, target.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        yield cooler.Cooler(str(target))


def annotations(path: str | Path) -> pd.DataFrame:
    """Read a unified CSV or the supplied Supplementary Tables 4/5/6."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        out = pd.read_csv(path)
    else:
        sheets = []
        for number, kind in ((4, "OPCID"), (5, "CHIN"), (6, "CHID")):
            table = pd.read_excel(path, sheet_name=f"Supplementary Table {number}")
            table.columns = table.columns.str.strip().str.lower()
            table["type"] = kind
            table["id"] = table.get(f"{kind.lower()}_id", pd.Series(
                [f"{kind}_{i}" for i in range(len(table))]))
            sheets.append(table)
        out = pd.concat(sheets, ignore_index=True)
    out.columns = out.columns.str.strip().str.lower()
    if "chr" in out and "chrom" not in out:
        out = out.rename(columns={"chr": "chrom"})
    for col in ("start", "end", "type"):
        if col not in out:
            raise ValueError(f"Annotation file lacks {col}: {path}")
    out = out.copy()
    out["start"] = pd.to_numeric(out["start"], errors="raise").astype(int)
    out["end"] = pd.to_numeric(out["end"], errors="raise").astype(int)
    out["type"] = out["type"].astype(str).str.upper().str.strip()
    out = out[out.type.isin(TYPES) & (out.end > out.start)].copy()
    midpoint = (out.start + out.end) // 2
    if "center" not in out:
        out["center"] = midpoint
    else:
        out["center"] = pd.to_numeric(out.center, errors="coerce").fillna(midpoint).astype(int)
    if "chrom" not in out:
        out["chrom"] = ""
    if "id" not in out:
        out["id"] = [f"structure_{i}" for i in range(len(out))]
    out["length_bp"] = out.end - out.start
    return out[["id", "type", "chrom", "start", "end", "center", "length_bp"]].reset_index(drop=True)


def chrom_name(c, requested: str | None = None) -> str:
    names = list(c.chromnames)
    if requested and requested in names:
        return requested
    if len(names) == 1 and (not requested or str(requested).lower() in
                            ("nan", "", "mg1655", "nc_000913.3")):
        return names[0]
    raise ValueError(f"Chromosome {requested!r} not found; available: {names}")


def annotation_chrom_mask(table: pd.DataFrame, chrom: str) -> pd.Series:
    """Match the MG1655 short name and its NC_000913.3 RefSeq accession."""
    aliases = {"MG1655", "NC_000913.3"}
    accepted = aliases if chrom in aliases else {chrom}
    return table.chrom.isin(accepted | {""})


def window(c, chrom: str, center_bp: int, size_bp: int, output_bins: int = 128) -> np.ndarray:
    """Read a centered cis window, with zero padding at chromosome edges."""
    chrom = chrom_name(c, chrom)
    bin_size = int(c.binsize)
    n = max(1, int(round(size_bp / bin_size)))
    lo, hi = center_bp // bin_size - n // 2, center_bp // bin_size - n // 2 + n
    extent = int(c.extent(chrom)[1] - c.extent(chrom)[0])
    a, b = max(0, lo), min(extent, hi)
    result = np.zeros((n, n), dtype=np.float32)
    if b > a:
        block = c.matrix(balance=False, sparse=True).fetch(
            (chrom, a * bin_size, b * bin_size)).toarray().astype(np.float32)
        off = a - lo
        result[off:off + b - a, off:off + b - a] = block
    if n != output_bins:
        # Legacy linear interpolation retained for frozen historical analyses.
        # It samples intensities and does not preserve total contact counts.
        # Use area_window for an explicit count-conserving measurement.
        from scipy.ndimage import zoom

        result = zoom(result, output_bins / n, order=1)
        result = result[:output_bins, :output_bins]
        if result.shape != (output_bins, output_bins):
            padded = np.zeros((output_bins, output_bins), np.float32)
            padded[:result.shape[0], :result.shape[1]] = result
            result = padded
    return (result + result.T) / 2


def area_resample(mat: np.ndarray, output_bins: int) -> np.ndarray:
    """Integrate counts over output pixel areas instead of sampling pixels.

    Integer downsampling sums disjoint blocks. Fractional boundaries distribute
    each native count by its area overlap, preserving the total and symmetry.
    Output values are counts, so normalization must occur after this operation.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if (mat.ndim != 2 or mat.shape[0] != mat.shape[1] or not len(mat) or
            not np.isfinite(mat).all()):
        raise ValueError("Area resampling requires a finite nonempty square matrix")
    if (isinstance(output_bins, bool) or not isinstance(output_bins, (int, np.integer)) or
            output_bins < 1):
        raise ValueError("output_bins must be a positive integer")
    n = len(mat)
    if n == output_bins:
        return mat.copy()
    if n % output_bins == 0:
        factor = n // output_bins
        return mat.reshape(output_bins, factor, output_bins, factor).sum(axis=(1, 3))
    edges = np.linspace(0, n, output_bins + 1)
    native = np.arange(n, dtype=float)
    overlap = np.maximum(0, np.minimum(edges[1:, None], native[None, :] + 1) -
                         np.maximum(edges[:-1, None], native[None, :]))
    return overlap @ mat @ overlap.T


def area_window(c, chrom: str, center_bp: int, size_bp: int,
                output_bins: int = 128) -> np.ndarray:
    """Read the same padded native window, then sum counts into output areas."""
    native_bins = max(1, int(round(size_bp / int(c.binsize))))
    native = window(c, chrom, center_bp, size_bp, native_bins)
    return area_resample(native, output_bins)


def normalize(mat: np.ndarray) -> np.ndarray:
    """Distance-normalized log contacts, retaining local shape rather than depth."""
    mat = np.log1p(np.maximum(np.asarray(mat, dtype=np.float32), 0))
    n = len(mat)
    expected = np.empty(n, dtype=np.float32)
    for d in range(n):
        diagonal = np.diagonal(mat, offset=d)
        expected[d] = np.median(diagonal) if len(diagonal) else 0
    distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    residual = mat - expected[distance]
    upper = residual[np.triu_indices(n, k=2)]
    scale = np.median(np.abs(upper - np.median(upper))) * 1.4826
    return np.clip(residual / max(float(scale), 0.1), -5, 5).astype(np.float32)


def upper_correlation(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.triu_indices(len(a), k=2)
    x, y = a[mask], b[mask]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def save_heatmap(mat: np.ndarray, path: Path, title: str, vmin=-2, vmax=4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    image = ax.imshow(mat, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set(title=title, xlabel="Genomic bin", ylabel="Genomic bin")
    fig.colorbar(image, ax=ax, label="Normalized contact")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
