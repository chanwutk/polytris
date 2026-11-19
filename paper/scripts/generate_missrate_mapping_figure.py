#!/usr/bin/env python3
"""Generate the TikZ figure showing the MissRate -> \\bar{G}^{\\bar{M}} mapping.

Reads accuracy.npy and sample_rates.json from a track_rate artifact
directory, crops a tile region (default 2 rows x 3 columns), and emits
a TikZ figure with MissRate matrices on the left and derived
\\bar{G}^{\\bar{M}} matrices on the right.

Source artifact (default): jnc7 / bytetrackcython_60
Crop region (default):     rows 9..10, columns 10..12
"""

import argparse
import ast
import json
import struct
import sys
from pathlib import Path


def load_npy(path: Path):
    """Minimal .npy loader using only the stdlib."""
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a .npy file")
        major, _ = f.read(2)
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header = ast.literal_eval(f.read(header_len).decode("latin1"))
        shape = header["shape"]
        descr = header["descr"]
        fmt_map = {
            "<f4": "f", "<f8": "d",
            "<i4": "i", "<i8": "q",
            "<u4": "I",
        }
        ch = fmt_map.get(descr)
        if ch is None:
            raise ValueError(f"unsupported dtype {descr}")
        n = 1
        for d in shape:
            n *= d
        data = struct.unpack(f"<{ch * n}", f.read(struct.calcsize(ch) * n))
    return shape, data


def idx3(shape, a, b, c):
    return (a * shape[1] + b) * shape[2] + c


def extract_missrate_crop(accuracy_shape, accuracy_data, gap_indices,
                          r0, c0, h, w):
    """Return {gap_index: [[missrate%]]} for the crop."""
    result = {}
    for gi in gap_indices:
        mat = []
        for i in range(r0, r0 + h):
            row = []
            for j in range(c0, c0 + w):
                acc = accuracy_data[idx3(accuracy_shape, gi, i, j)]
                row.append(round(100 * round(1 - acc, 2)))
            mat.append(row)
        result[gi] = mat
    return result


def derive_u(accuracy_shape, accuracy_data, sample_rates, M, r0, c0, h, w):
    """Derive \\bar{G}^{\\bar{M}} for the crop using the paper equation."""
    mat = []
    for i in range(r0, r0 + h):
        row = []
        for j in range(c0, c0 + w):
            best = 1
            for gi, g in enumerate(sample_rates):
                # Use the same two-decimal fractional miss rate represented by the displayed percentages.
                miss = round(1 - accuracy_data[idx3(accuracy_shape, gi, i, j)], 2)
                if miss <= M:
                    best = g
            row.append(best)
        mat.append(row)
    return mat


def color_for_missrate(pct):
    """Return a TikZ fill color string for a mistrack-rate percentage."""
    t = min(pct / 100.0, 1.0)
    if t < 0.25:
        return f"green!{round(60 * (1 - t / 0.25))}!white"
    if t < 0.50:
        s = (t - 0.25) / 0.25
        return f"yellow!{round(50 + 20 * s)}!white"
    return f"red!{round(30 + 40 * min((t - 0.50) / 0.50, 1.0))}!white"


def color_for_gap(g):
    """Return a TikZ fill color string for a gap value."""
    mapping = {
        1: "red!50!white",
        2: "orange!45!white",
        4: "yellow!50!white",
        8: "green!40!white",
        16: "teal!35!white",
    }
    return mapping.get(g, "black!10")


def r(v):
    """Round a coordinate to two decimal places."""
    return f"{v:.2f}"


def emit_matrix(lines, mat, x0, y0, cell, color_fn, fmt_fn,
                value_label=None, value_prefix=None,
                frame_image=None):
    """Append a matrix with a lower-left value badge and optional prefix."""
    h = len(mat)
    w = len(mat[0])
    mat_w = w * cell
    mat_h = h * cell
    if frame_image:
        cx = x0 + mat_w / 2
        cy = y0 - mat_h / 2
        lines.append(
            f"  \\node[inner sep=0pt] at ({r(cx)},{r(cy)})"
            f" {{\\includegraphics[width={r(mat_w)}cm, height={r(mat_h)}cm]{{{frame_image}}}}};"
        )
    opacity = ", fill opacity=0.35" if frame_image else ""
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            cx = x0 + j * cell + cell / 2
            cy = y0 - i * cell - cell / 2
            # Keep the lower-left tile's statistic clear of the corner badge.
            if value_label and i == h - 1 and j == 0:
                cx += 0.14
            fill = color_fn(val)
            text = fmt_fn(val)
            lines.append(
                f"  \\fill[{fill}{opacity}, draw=white, line width=0.6pt]"
                f" ({r(x0 + j * cell)},{r(y0 - i * cell)})"
                f" rectangle ({r(x0 + (j + 1) * cell)},{r(y0 - (i + 1) * cell)});"
            )
            lines.append(
                f"  \\node[celltext] at ({r(cx)},{r(cy)}) {{\\contour{{black}}{{{text}}}}};"
            )
    if value_label:
        badge_y = y0 - mat_h + 0.26
        badge_name = f"parameter{len(lines)}"
        lines.append(
            f"  \\node[parambadge] ({badge_name}) at ({r(x0)},{r(badge_y)}) {{{value_label}}};"
        )
        if value_prefix:
            # Share the numeral's baseline rather than centering unequal-height math labels.
            lines.append(
                f"  \\node[gridlabel, anchor=base east] at ([xshift=-2pt]{badge_name}.base west)"
                f" {{{value_prefix}}};"
            )


def generate_tikz(missrate_g2, missrate_g4, missrate_g8,
                  u_04, u_06, u_08,
                  frame_image=None):
    cell = 0.72
    rows = len(missrate_g2)
    cols = len(missrate_g2[0])
    mat_h = rows * cell
    mat_w = cols * cell

    h_dots_gap = 0.6
    grid_gap = 0.4
    arrow_gap = 0.7
    row_label_pad = 0.4

    left_x = 0
    mid_x = mat_w + grid_gap
    right_x = 2 * (mat_w + grid_gap)

    top_y0 = 0
    bot_y0 = -mat_h - arrow_gap

    lines = []
    lines.append("\\begin{tikzpicture}[x=1cm,y=1cm]")
    lines.append("  \\tikzstyle{celltext}=[font=\\sffamily\\large, text=white, inner sep=0pt]")
    lines.append("  \\tikzstyle{axislabel}=[font=\\LARGE\\bfseries]")
    lines.append("  \\tikzstyle{gridlabel}=[font=\\scriptsize, inner sep=1pt]")
    lines.append("  \\tikzstyle{parambadge}=[circle, draw=black, fill=white, line width=0.6pt, minimum size=0.40cm, inner sep=0pt, font=\\scriptsize]")
    lines.append("  \\tikzstyle{dotstext}=[font=\\bfseries\\Large, inner sep=0pt]")

    miss_fmt = lambda v: f"{v}"
    gap_fmt = lambda v: f"{v}"

    emit_matrix(lines, missrate_g2, left_x, top_y0,
                cell, color_for_missrate, miss_fmt,
                value_label=r"$2$", value_prefix=r"$\gamma=$",
                frame_image=frame_image)
    emit_matrix(lines, missrate_g4, mid_x, top_y0,
                cell, color_for_missrate, miss_fmt,
                value_label=r"$4$",
                frame_image=frame_image)
    emit_matrix(lines, missrate_g8, right_x, top_y0,
                cell, color_for_missrate, miss_fmt,
                value_label=r"$8$",
                frame_image=frame_image)

    emit_matrix(lines, u_04, left_x, bot_y0,
                cell, color_for_gap, gap_fmt,
                value_label=r"$.4$", value_prefix=r"$\bar{M}=$",
                frame_image=frame_image)
    emit_matrix(lines, u_06, mid_x, bot_y0,
                cell, color_for_gap, gap_fmt,
                value_label=r"$.6$",
                frame_image=frame_image)
    emit_matrix(lines, u_08, right_x, bot_y0,
                cell, color_for_gap, gap_fmt,
                value_label=r"$.8$",
                frame_image=frame_image)

    hdots_before = left_x - h_dots_gap / 2
    hdots_after = right_x + mat_w + h_dots_gap / 2

    for row_y_center in (top_y0 - mat_h / 2, bot_y0 - mat_h / 2):
        for hx in (hdots_before, hdots_after):
            lines.append(
                f"  \\node[dotstext] at ({r(hx)},{r(row_y_center)}) {{$\\boldsymbol{{\\cdots}}$}};"
            )

    row_label_x = hdots_before - row_label_pad
    lines.append(
        f"  \\node[axislabel, anchor=east, align=right] at ({r(row_label_x)},{r(top_y0 - mat_h / 2)})"
        f" {{$\\boldsymbol{{\\mathrm{{MissRate}}^{{(\\gamma)}}}}$ \\\\ \\normalsize(percent)}};"
    )
    lines.append(
        f"  \\node[axislabel, anchor=east] at ({r(row_label_x)},{r(bot_y0 - mat_h / 2)})"
        f" {{$\\boldsymbol{{\\bar{{G}}^{{\\bar{{M}}}}}}$}};"
    )

    arrow_x = mid_x + mat_w / 2
    arrow_top_y = top_y0 - mat_h - 0.1
    arrow_bot_y = bot_y0 + 0.1
    lines.append(
        f"  \\draw[-stealth, line width=2pt]"
        f" ({r(arrow_x)},{r(arrow_top_y)})"
        f" -- ({r(arrow_x)},{r(arrow_bot_y)})"
        f" node[midway, anchor=west, xshift=4pt, font=\\large]"
        f" {{$\\bar{{g}}^{{\\bar{{M}}}}_{{i,j}} = \\max\\{{\\gamma \\mid \\mathrm{{MissRate}}^{{(\\gamma)}}_{{i,j}} \\leq \\bar{{M}}\\}}$}};"
    )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/data/cwkt/data/polyis-cache/jnc7/indexing/track_rate/bytetrackcython_60"),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "figures" / "generated" / "missrate_to_g_mapping.tex",
    )
    parser.add_argument(
        "--frame-image",
        type=str,
        default=None,
        help="Path to a cropped video frame PNG to overlay behind each matrix "
             "(relative to the LaTeX build directory).",
    )
    parser.add_argument("--row-start", type=int, default=9)
    parser.add_argument("--col-start", type=int, default=10)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    sample_rates = json.loads((args.artifact_dir / "sample_rates.json").read_text())
    acc_shape, acc_data = load_npy(args.artifact_dir / "accuracy.npy")

    r0, c0, h, w = args.row_start, args.col_start, args.rows, args.cols
    gi_map = {g: i for i, g in enumerate(sample_rates)}

    missrate_g2 = extract_missrate_crop(
        acc_shape, acc_data, [gi_map[2]], r0, c0, h, w
    )[gi_map[2]]
    missrate_g4 = extract_missrate_crop(
        acc_shape, acc_data, [gi_map[4]], r0, c0, h, w
    )[gi_map[4]]
    missrate_g8 = extract_missrate_crop(
        acc_shape, acc_data, [gi_map[8]], r0, c0, h, w
    )[gi_map[8]]

    u_04 = derive_u(acc_shape, acc_data, sample_rates, 0.4, r0, c0, h, w)
    u_06 = derive_u(acc_shape, acc_data, sample_rates, 0.6, r0, c0, h, w)
    u_08 = derive_u(acc_shape, acc_data, sample_rates, 0.8, r0, c0, h, w)

    tikz = generate_tikz(missrate_g2, missrate_g4, missrate_g8,
                         u_04, u_06, u_08,
                         frame_image=args.frame_image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tikz + "\n")
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
