#!/usr/bin/env python3
"""Cutoff-free invariance projection of phonopy force constants (twisted / moiré post-processing step).

Projects the full supercell FORCE_CONSTANTS of a FINAL_RESULTS*/<cell>/ directory onto the subspace that
satisfies, exactly, the three invariances the true force constants must obey:
  * translational ASR          sum_j Phi_ij = 0
  * Born-Huang rotational      sum_j Phi_ij^{ab} r_ij^c - Phi_ij^{ac} r_ij^b = 0   (per atom i)
  * Huang bracket symmetry     [ab,cd] = [cd,ab],  [ab,cd] = sum_ij Phi_ij^{ab} r_ij^c r_ij^d   (zero-stress form)
Finite-displacement supercell FCs violate the last two (truncation + force noise amplified by the r, r^2 lever
arms), which shows up as a spurious negative q^2 term of the flexural ZA branch near Gamma (-0.04..-0.07 THz dips
at |q| ~ 0.06 1/A in the grid-tight twisted cells, 2026-10-24). hiphive cannot be used on moiré cells (its real-space
cutoff truncates the interlayer block); this projection has no cutoff.

Formulation (must be consistent with phonopy's dynamical matrix, otherwise boundary pairs are image-ambiguous):
  * unknowns = compact Phi(i in primitive, j in supercell), parameterised by permutation/translation orbits so that
    Phi_ij = Phi_ji^T holds exactly;
  * pair vectors = phonopy's multiplicity-averaged shortest vectors (the ones D(q) is interpolated with);
  * one minimum-norm (uniform metric) solve.  dPhi is ~1e-4 eV/A^2 (FC rms ~1); optical modes move < 0.002 THz.
Pins (default on): (a) the four rigid-layer 3x3 Gamma-block sums (K, breathing sums) and (b) the 3 lowest optical
Gamma modes as exact eigenvectors with unchanged eigenvalues (shear pair + breathing).  Without them the min-norm
correction leaks into the interlayer block and the Gamma shear pair goes imaginary.  The pins preserve whatever the
raw Gamma block is, so the raw statics must be trusted there (grid-tight INCAR; K verified against the sliding scan).
What it cannot fix: q^4-level (bending-rigidity) errors (MoTe2 m2 keeps a -0.06 THz lobe), and it needs a supercell
>= ~14 A (the 42-atom 1x1x1 m1 cell is too small: bands move 0.03 THz).

Per cell it writes: plain_phonopy/{FORCE_CONSTANTS,band.yaml,band.pdf} (raw copies, made once), the projected
FORCE_CONSTANTS, regenerated band.yaml/band.pdf (same Gamma-K-M-Gamma path and q-count as before) and
invariance_projection.json (residuals, dPhi, K / Gamma modes / ZA / mesh before and after).  Re-running always
starts from plain_phonopy/FORCE_CONSTANTS, so it is idempotent.  total_dos.dat is NOT regenerated — rerun
phonopy/write_phonon_dos.py afterwards if the cell has one.

Run from the nequix uv env (needs phonopy >= 2.x dense smallest-vectors format + scipy):
    cd ~/twist/nequix && uv run python ../workflow/phonopy/project_invariant_force_constants.py \
        ../workflow/FINAL_RESULTS_TWISTED_GRIDTIGHT/MoS2_twist_m1_near0 [more dirs] [--dry-run]
Validated 2026-10-24 on the 20 hiphive-refit FINAL_RESULTS_BARE_PBE_SUBTRACT cells (clears 14/20 raw ZA dips with less
collateral change than hiphive) and on the 4 grid-tight twisted cells; see twist/logs/2026-10-24.md.
"""
import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

warnings.filterwarnings("ignore")
import phonopy  # noqa: E402
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS  # noqa: E402
from phonopy.harmonic.force_constants import compact_fc_to_full_fc  # noqa: E402

WAYPOINTS = np.array([[0, 0, 0], [1 / 3, 1 / 3, 0], [0.5, 0, 0], [0, 0, 0]])
LABELS = ["Γ", "K", "M", "Γ"]


def project(ph, fc_in, *, pin_layers=True, pin_gamma_exact=3, use_rot=True, use_huang=True):
    """Return (fc_full_projected, info) for a phonopy object and its raw FORCE_CONSTANTS (full or compact)."""
    prim, sc = ph.primitive, ph.supercell
    Np, Ns = len(prim), len(sc)
    p2s = np.array(prim.p2s_map)
    s2p = np.array(prim.s2p_map)
    p2p = prim.p2p_map
    fc_c = fc_in[p2s] if fc_in.shape[0] == Ns else fc_in
    assert fc_c.shape == (Np, Ns, 3, 3), fc_c.shape
    fc_full_in = compact_fc_to_full_fc(prim, fc_c)

    # multiplicity-averaged pair vectors r_ij = r_j - r_i (phonopy convention, dense format)
    svecs, multi = prim.get_smallest_vectors()
    svecs_c = np.array(svecs) @ np.array(prim.cell)
    R1 = np.zeros((Np, Ns, 3))
    R2 = np.zeros((Np, Ns, 3, 3))
    mult = np.zeros((Np, Ns), int)
    for j in range(Ns):
        for i in range(Np):
            m, s = multi[j, i]
            v = svecs_c[s:s + m]
            mult[i, j] = m
            R1[i, j] = v.mean(0)
            R2[i, j] = np.einsum("ka,kb->ab", v, v) / m
    Lsc = np.array(sc.cell)
    pos = np.array(sc.positions)
    frac = pos @ np.linalg.inv(Lsc)
    zmed = np.median(pos[:, 2])
    layer_s = (pos[:, 2] > zmed).astype(int)
    layer_p = layer_s[p2s]
    inter = layer_p[:, None] != layer_s[None, :]

    def idx(i, j, al, be):
        return ((i * Ns + j) * 3 + al) * 3 + be

    def atom_at(target):
        df = frac - (target @ np.linalg.inv(Lsc))
        df -= np.round(df)
        dist = np.linalg.norm(df @ Lsc, axis=1)
        k = int(dist.argmin())
        assert dist[k] < 1e-4, dist[k]
        return k

    # permutation + supercell-translation orbits -> parameters
    parent = np.arange(Np * Ns * 9)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    partner_k = np.zeros((Np, Ns), int)
    for j in range(Ns):
        Lj = pos[j] - pos[s2p[j]]
        for i in range(Np):
            partner_k[i, j] = atom_at(pos[p2s[i]] - Lj)
    for i in range(Np):
        for j in range(Ns):
            ip, k = p2p[s2p[j]], partner_k[i, j]
            for al in range(3):
                for be in range(3):
                    rx, ry = find(idx(i, j, al, be)), find(idx(ip, k, be, al))
                    if rx != ry:
                        parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(x) for x in range(Np * Ns * 9)])
    uniq, inv = np.unique(roots, return_inverse=True)
    npar = len(uniq)
    M = sp.csr_matrix((np.ones(Np * Ns * 9), (np.arange(Np * Ns * 9), inv)), shape=(Np * Ns * 9, npar))
    x0 = fc_c.reshape(-1)
    p0 = (M.T @ x0) / np.asarray(M.sum(0)).ravel()
    sym_err = float(np.abs(M @ p0 - x0).max())

    rows, cols, vals, kinds = [], [], [], []
    c = 0
    J = np.arange(Ns)
    for i in range(Np):  # translational ASR
        for al in range(3):
            for be in range(3):
                rows.append(np.full(Ns, c)); cols.append(idx(i, J, al, be)); vals.append(np.ones(Ns)); kinds.append("t"); c += 1
    if use_rot:  # Born-Huang rotational, per atom
        for i in range(Np):
            for al in range(3):
                for be in range(3):
                    for ga in range(be + 1, 3):
                        rows.append(np.full(2 * Ns, c))
                        cols.append(np.concatenate([idx(i, J, al, be), idx(i, J, al, ga)]))
                        vals.append(np.concatenate([R1[i, :, ga], -R1[i, :, be]])); kinds.append("r"); c += 1
    II, JJ = np.meshgrid(np.arange(Np), J, indexing="ij")
    II, JJ = II.ravel(), JJ.ravel()
    if use_huang:  # Huang bracket symmetry
        pairs = [(al, be) for al in range(3) for be in range(3)]
        for p in range(9):
            for q in range(p + 1, 9):
                (al, be), (ga, de) = pairs[p], pairs[q]
                rows.append(np.full(2 * Np * Ns, c))
                cols.append(np.concatenate([idx(II, JJ, al, be), idx(II, JJ, ga, de)]))
                vals.append(np.concatenate([R2[II, JJ, ga, de], -R2[II, JJ, al, be]])); kinds.append("h"); c += 1
    n_inv = c
    if pin_layers:  # rigid-layer Gamma block sums
        for la in (0, 1):
            for lb in (0, 1):
                si, sj = np.where(layer_p == la)[0], np.where(layer_s == lb)[0]
                I2, J2 = np.meshgrid(si, sj, indexing="ij")
                I2, J2 = I2.ravel(), J2.ravel()
                for al in range(3):
                    for be in range(3):
                        rows.append(np.full(len(I2), c)); cols.append(idx(I2, J2, al, be)); vals.append(np.ones(len(I2))); kinds.append("p"); c += 1
    if pin_gamma_exact:  # dD(Gamma) e_m = 0 for the lowest optical Gamma modes
        ph.force_constants = fc_full_in
        ph.run_qpoints([[0, 0, 0]], with_eigenvectors=True)
        E = np.real(ph.get_qpoints_dict()["eigenvectors"][0])
        mass_p = np.array(prim.masses)
        jp_of = np.array([p2p[s2p[j]] for j in range(Ns)])
        for m in range(3, 3 + pin_gamma_exact):
            em = E[:, m].reshape(Np, 3)
            for ip in range(Np):
                for al in range(3):
                    coef = em[jp_of, :] / np.sqrt(mass_p[ip] * mass_p[jp_of])[:, None]
                    rows.append(np.full(3 * Ns, c))
                    cols.append(np.concatenate([idx(ip, J, al, be) for be in range(3)]))
                    vals.append(np.concatenate([coef[:, be] for be in range(3)])); kinds.append("p"); c += 1
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(c, Np * Ns * 9))
    kinds = np.array(kinds)
    rhs = np.zeros(c)
    rhs[kinds == "p"] = C[kinds == "p"] @ x0
    A = (C @ M).tocsr()
    G = (A @ A.T).toarray()
    y = np.linalg.lstsq(G, rhs - A @ p0, rcond=1e-13)[0]
    x1 = M @ (p0 + A.T @ y)
    fc_full1 = compact_fc_to_full_fc(prim, x1.reshape(Np, Ns, 3, 3))

    def resid(x):
        r = C[kinds != "p"] @ x
        k = kinds[kinds != "p"]
        return {kk: float(np.abs(r[k == kk]).max()) if (k == kk).any() else 0.0 for kk in "trh"}

    d = (x1 - x0).reshape(Np, Ns, 9)
    info = {
        "Np": Np, "Ns": Ns, "n_params": int(npar), "n_constraints": int(c), "n_invariance_rows": int(n_inv),
        "n_pairs_multi_image": int((mult > 1).sum()), "perm_sym_err_input": sym_err,
        "resid_input": resid(x0), "resid_projected": resid(x1), "fc_rms": float(np.sqrt(np.mean(x0 ** 2))),
        "dPhi_rms": float(np.sqrt(np.mean(d ** 2))), "dPhi_max": float(np.abs(d).max()),
        "dPhi_rms_inter": float(np.sqrt(np.mean(d[inter] ** 2))), "dPhi_rms_intra": float(np.sqrt(np.mean(d[~inter] ** 2))),
        "layer_p": layer_p, "layer_s": layer_s, "p2s": p2s,
    }
    return fc_full_in, fc_full1, info


def evaluate(ph, fc_full, info, mesh):
    """K_prim, lowest Gamma modes, ZA scans with mode character, path/mesh minima."""
    layer_p, layer_s, p2s = info["layer_p"], info["layer_s"], info["p2s"]
    ph.force_constants = fc_full
    Np = len(ph.primitive)
    mass = np.array(ph.primitive.masses)
    K = float(fc_full[p2s][np.ix_(layer_p == 1, layer_s == 1)][:, :, 0, 0].sum())
    g = ph.get_frequencies([0, 0, 0])
    Bcell = 2 * np.pi * np.linalg.inv(np.array(ph.primitive.cell)).T

    def character(q, band=0):
        ph.run_qpoints([q], with_eigenvectors=True)
        dd = ph.get_qpoints_dict()
        v = dd["eigenvectors"][0][:, band].reshape(Np, 3) / np.sqrt(mass)[:, None]
        n2 = np.sum(np.abs(v) ** 2)
        U = [v[layer_p == L].mean(0) for L in (0, 1)]
        NL = [(layer_p == L).sum() for L in (0, 1)]
        return {"f": float(dd["frequencies"][0][band]), "z_frac": float(np.sum(np.abs(v[:, 2]) ** 2) / n2),
                "rigid_frac": float(sum(NL[L] * np.sum(np.abs(U[L]) ** 2) for L in (0, 1)) / n2),
                "inphase_frac": float(np.sum(np.abs(U[0] + U[1]) ** 2) / (np.sum(np.abs(U[0] + U[1]) ** 2) + np.sum(np.abs(U[0] - U[1]) ** 2) + 1e-30))}

    res = {"K_prim": K, "gamma6": [float(v) for v in g[:6]]}
    for name, qs, xmax in (("GM", lambda x: [x, 0, 0], 0.2), ("GK", lambda x: [x, x, 0], 0.12)):
        xs = np.linspace(0, xmax, 61)
        Q = np.array([qs(x) for x in xs])
        ph.run_qpoints(Q)
        fr = ph.get_qpoints_dict()["frequencies"][:, 0]
        qa = np.linalg.norm(Q @ Bcell, axis=1)
        w2 = np.sign(fr) * fr ** 2
        m = (qa > 0) & (qa < 0.25)
        aa, bb = np.linalg.lstsq(np.vstack([qa[m] ** 2, qa[m] ** 4]).T, w2[m], rcond=None)[0]
        res[name] = {"min": float(fr.min()), "q_at_min_invA": float(qa[fr.argmin()]), "a_q2": float(aa), "b_q4": float(bb),
                     "char_at_min": character(Q[fr.argmin()])}
    ph.run_mesh([mesh, mesh, 1])
    fm = ph.get_mesh_dict()["frequencies"]
    res["mesh"] = [mesh, mesh, 1]
    res["mesh_min"] = float(fm.min())
    res["mesh_n_neg"] = int((fm < -1e-3).sum())
    return res


def band_path(ph, seg_n):
    paths = [np.array([WAYPOINTS[i] + (WAYPOINTS[i + 1] - WAYPOINTS[i]) * t / (seg_n[i] - 1) for t in range(seg_n[i])]) for i in range(3)]
    ph.run_band_structure(paths, labels=LABELS, path_connections=[True, True, False])
    return np.vstack(ph.get_band_structure_dict()["frequencies"])


def process(d: Path, args) -> dict:
    raw_dir = d / "plain_phonopy"
    if not (raw_dir / "FORCE_CONSTANTS").exists():
        if args.dry_run:
            print(f"  [dry-run] would create {raw_dir} from the current FORCE_CONSTANTS/band.*")
        else:
            raw_dir.mkdir(exist_ok=True)
            for f in ("FORCE_CONSTANTS", "band.yaml", "band.pdf"):
                if (d / f).exists():
                    shutil.copy2(d / f, raw_dir / f)
    raw_fc_file = raw_dir / "FORCE_CONSTANTS" if (raw_dir / "FORCE_CONSTANTS").exists() else d / "FORCE_CONSTANTS"
    ph = phonopy.load(str(d / "phonopy.yaml"), produce_fc=False, log_level=0)
    fc_in = parse_FORCE_CONSTANTS(str(raw_fc_file))
    fc_full_in, fc_full1, info = project(ph, fc_in, pin_layers=not args.no_pin_layers, pin_gamma_exact=args.pin_gamma_exact,
                                         use_rot=not args.no_rot, use_huang=not args.no_huang)
    seg_n = yaml.safe_load((d / "band.yaml").read_text())["segment_nqpoint"] if (d / "band.yaml").exists() else [51, 51, 51]
    r0 = evaluate(ph, fc_full_in, info, args.mesh)
    fb0 = band_path(ph, seg_n)
    r1 = evaluate(ph, fc_full1, info, args.mesh)
    fb1 = band_path(ph, seg_n)
    hi = fb0 > 1.0
    r1["band_rmse_vs_raw_all"] = float(np.sqrt(np.mean((fb1 - fb0) ** 2)))
    r1["band_rmse_vs_raw_above1THz"] = float(np.sqrt(np.mean((fb1[hi] - fb0[hi]) ** 2)))
    r1["band_maxabs_vs_raw_above1THz"] = float(np.abs(fb1[hi] - fb0[hi]).max())
    r1["band_min"] = float(fb1.min())
    r0["band_min"] = float(fb0.min())
    summary = {"cell": d.name, "raw_fc": str(raw_fc_file.relative_to(d)), "method": "cutoff-free translational+rotational+Huang projection, "
               "phonopy-consistent multiplicity vectors, orbit parameters, uniform min-norm (phonopy/project_invariant_force_constants.py)",
               "pin_layers": not args.no_pin_layers, "pin_gamma_exact": args.pin_gamma_exact, "rotational": not args.no_rot, "huang": not args.no_huang,
               **{k: v for k, v in info.items() if k not in ("layer_p", "layer_s", "p2s")}, "raw": r0, "projected": r1}
    print(f"== {d.name}: Np={info['Np']} Ns={info['Ns']} params={info['n_params']} constraints={info['n_constraints']} "
          f"resid rot {info['resid_input']['r']:.2e}->{info['resid_projected']['r']:.1e} Huang {info['resid_input']['h']:.2e}->{info['resid_projected']['h']:.1e}")
    for lab, r in (("raw      ", r0), ("projected", r1)):
        ch = r["GM"]["char_at_min"]
        print(f"   {lab}: K={r['K_prim']:+.4f} Γ6={np.round(r['gamma6'], 3)} lowest G-M min {r['GM']['min']:+.4f} THz @ {r['GM']['q_at_min_invA']:.3f}/A "
              f"[z {ch['z_frac']:.2f} inphase {ch['inphase_frac']:.2f}] a={r['GM']['a_q2']:+.3f} mesh min {r['mesh_min']:+.4f} ({r['mesh_n_neg']} neg)")
    print(f"   dPhi rms {info['dPhi_rms']:.1e} (FC rms {info['fc_rms']:.2f}); band RMSE vs raw >1 THz {r1['band_rmse_vs_raw_above1THz']:.4f} (max {r1['band_maxabs_vs_raw_above1THz']:.4f})")
    if not args.dry_run:
        write_FORCE_CONSTANTS(fc_full1, filename=str(d / "FORCE_CONSTANTS"))
        ph.force_constants = fc_full1
        band_path(ph, seg_n)
        ph.write_yaml_band_structure(filename=str(d / "band.yaml"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = ph.plot_band_structure()
        fig.savefig(d / "band.pdf")
        plt.close("all")
        (d / "invariance_projection.json").write_text(json.dumps(summary, indent=1))
        if (d / "total_dos.dat").exists():
            print(f"   NOTE: {d / 'total_dos.dat'} is stale — rerun phonopy/write_phonon_dos.py {d}")
        print(f"   wrote FORCE_CONSTANTS, band.yaml, band.pdf, invariance_projection.json (raw kept in plain_phonopy/)")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="FINAL_RESULTS*/<cell> dirs (phonopy.yaml + FORCE_CONSTANTS)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--no-pin-layers", action="store_true", help="do not pin the rigid-layer Gamma block (diagnostic)")
    ap.add_argument("--pin-gamma-exact", type=int, default=3, help="number of lowest optical Gamma modes kept exact (default 3; 0 = off)")
    ap.add_argument("--no-rot", action="store_true", help="diagnostic: drop the rotational rule")
    ap.add_argument("--no-huang", action="store_true", help="diagnostic: drop the Huang conditions")
    ap.add_argument("--mesh", type=int, default=24, help="NxNx1 mesh for the negative-frequency check (default 24)")
    args = ap.parse_args()
    for d in args.dirs:
        process(d.resolve(), args)


if __name__ == "__main__":
    main()
