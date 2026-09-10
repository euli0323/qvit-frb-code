#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
direct_plot_en.py



Given a PSRFITS file plus a start/end time (in seconds since the beginning
of the data), directly plot the RFI-mitigated dynamic spectrum (narrowband
RFI removed).

Processing chain:
    1) Grab the sample range covering the [s, e] time window (read only the
       required SUBINT rows, so the whole multi-GB DATA column never has to
       be loaded into memory at once)
    2) Vectorized unpack
    3) Total-intensity synthesis: AABBCRCI -> 0.5*(AA+BB); IQUV -> take I
    4) bin_data (time/frequency decimation) + zap_rfi_freq (removes narrowband RFI)
    5) imshow (cmap=binary, origin=lower, vmin/vmax = mean +/- 3 sigma),
       with or without axes



Usage:
    python direct_plot_en.py -f in.fits -s 0.40 -e 0.48
    python direct_plot_en.py -f in.fits -s 0.40 -e 0.48 -o out.png --noaxis

Optional arguments (only needed when you want to override the defaults,
which are fine in most cases):
    -bs / -bf     time / frequency bin factor (default 16 / 16)
    --cmap        matplotlib colormap (default binary)
    --dpi         output figure resolution (default 180)
"""

import argparse
import gc
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from astropy.io import fits
from astropy.time import Time


# ---------------------------------------------------------------------------
# Core processing functions (vectorized versions)
# ---------------------------------------------------------------------------
def unpack_data_vectorized(data_grab, nbits, nchan, nsamp, npol):
    """Vectorized unpacking: process all samples and polarizations at once."""
    packed_dim = data_grab.shape[2]                       # nchan * nbits / 8
    data_2d = data_grab.reshape(-1, packed_dim).astype(np.uint8)
    bits = np.unpackbits(data_2d, axis=1)                 # (N, nchan*nbits)
    bits = bits.reshape(nsamp, npol, nchan, nbits)
    bits = bits[:, :, :, ::-1]                            # equivalent to fliplr
    bits_2d = bits.reshape(-1, nbits)
    packed = np.packbits(bits_2d, axis=1, bitorder='little')
    return packed.reshape(nsamp, npol, nchan)




def zap_rfi_freq(dat, nchan, bs, bf):
    """Remove narrowband RFI: replace bad channels with a window average."""
    spec = np.sum(dat, axis=1)
    std = np.std(dat, axis=1)
    mask = np.ones(nchan, dtype=bool)
    n0, n1 = 0, 1
    while n0 != n1:
        n0 = nchan - np.count_nonzero(mask)
        std_spec, mean_spec = np.std(spec[mask]), np.mean(spec[mask])
        std_std, mean_std = np.std(std[mask]), np.mean(std[mask])
        mask &= ~((np.abs(spec - mean_spec) > 3 * std_spec ) |
                  (np.abs(std - mean_std) > 3 * std_std ))
        n1 = nchan - np.count_nonzero(mask)
    dat_mean = np.mean(dat[mask, :])
    dat[~mask, :] = dat_mean


def bin_data(dat, num_bin, num_chn):
    """First bin along frequency (rows) by num_chn, then along time (columns)
    by num_bin."""
    m, _ = dat.shape
    weight = np.arange(0, m, num_chn)
    dat_plot = np.add.reduceat(dat, weight)
    _, n = dat_plot.shape
    weight = np.arange(0, n, num_bin)
    dat_plot = np.add.reduceat(dat_plot, weight, axis=1)
    return dat_plot


def pick_polarization_intensity(data_unpack, npol, pol_order):
    """
    Select total intensity according to the polarization mode:
      IQUV*         -> take I (idx 0)
      AABB* / other -> 0.5*(pol0 + pol1)
      NPOL == 1     -> unchanged
    Returns a float64 array of shape (nchan, nsamp).
    """
    if pol_order and pol_order.startswith('IQUV'):
        intensity = data_unpack[:, 0, :]
    elif npol >= 2:
        intensity = 0.5 * (data_unpack[:, 0, :] + data_unpack[:, 1, :])
    else:
        intensity = np.squeeze(data_unpack, axis=1)
    return intensity.T.astype(np.float64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Plot an RFI-removed FITS dynamic spectrum directly from '
                    'a start/end time'
    )
    p.add_argument('-f', '--input_file', required=True, help='PSRFITS file path')
    p.add_argument('-s', '--start_time', type=float, required=True,
                   help='Start time in seconds since the beginning of the data')
    p.add_argument('-e', '--end_time', type=float, required=True,
                   help='End time in seconds since the beginning of the data')
    p.add_argument('-o', '--output_file', default=None,
                   help='Output PNG path (default: <input>_<s>_<e>.png)')
    p.add_argument('-bs', '--bin_samp', type=int, default=None,
                   help='Time bin factor (default 4)')
    p.add_argument('-bf', '--bin_chn', type=int, default=None,
                   help='Frequency bin factor (default 16)')
    p.add_argument('--cmap', default='binary',
                   help='matplotlib colormap (default binary)')
    p.add_argument('--dpi', type=int, default=180, help='Output DPI (default 180)')
    p.add_argument('--noaxis', action='store_true',
                   help='Do not draw axes (bare image)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    in_file = args.input_file
    t_start = float(args.start_time)
    t_end = float(args.end_time)
    if t_end <= t_start:
        raise SystemExit('[error] end_time must be greater than start_time')

    t0 = time.time()

    # ---- Read FITS headers ----
    hdul = fits.open(in_file, memmap=True)
    h0 = hdul[0].header
    sd = hdul['SUBINT']
    sh = sd.header
    nbits = int(sh['NBITS'])
    nchan = int(sh['NCHAN'])
    nsblk = int(sh['NSBLK'])
    nsub = int(sh['NAXIS2'])
    npol = int(sh['NPOL'])
    tsamp = float(sh['TBIN'])                          # seconds per sample
    pol_order = sh.get('POL_TYPE', '') or ''
    stt_imjd = float(h0.get('STT_IMJD', 0))
    stt_smjd = float(h0.get('STT_SMJD', 0))
    mjd0 = stt_imjd + stt_smjd / 86400.0

    N_total = nsub * nsblk
    dur_total = N_total * tsamp

    s0 = int(np.floor(t_start / tsamp))
    s1 = int(np.ceil(t_end / tsamp))
    s0 = max(0, min(s0, N_total - 1))
    s1 = max(s0 + 1, min(s1, N_total))
    if t_start - s0 * tsamp > 1e-9:
        print(f'[note] start {t_start:.6f}s quantized to {s0*tsamp:.6f}s '
              f'(sample {s0})')
    if s1 * tsamp - t_end > 1e-9:
        print(f'[note] end {t_end:.6f}s quantized to {s1*tsamp:.6f}s '
              f'(sample {s1})')
    if s0 * tsamp > t_start - 1e-9 and s0 != 0:
        print(f'[warn] window clamped to file start 0.0s (requested '
              f'{t_start:.6f}s)')
    if s1 * tsamp < t_end and s1 == N_total:
        print(f'[warn] window clamped to file end {dur_total:.4f}s (requested '
              f'{t_end:.6f}s)')

    r0 = s0 // nsblk
    r1 = (s1 + nsblk - 1) // nsblk                    # includes the row holding s1-1
    nrows = r1 - r0

    print(f'file  = {in_file}')
    print(f'header= NBITS={nbits} NCHAN={nchan} NSBLK={nsblk} NAXIS2={nsub} '
          f'NPOL={npol} TBIN={tsamp}')
    print(f'POL_TYPE={pol_order}  total dur={dur_total:.4f}s  ({N_total} samp)')
    print(f'window start={t_start:.4f}s  -> sample {s0}')
    print(f'window end  ={t_end:.4f}s  -> sample {s1}')
    print(f'subint rows needed: r0={r0} r1={r1}  (loaded {nrows*nsblk} samp, '
          f'used {s1-s0})')

    # ---- Read only the required subint rows ----
    # The DATA column TDIM is like (1, NCHAN, NPOL, NSBLK) -- after slicing, the
    # shape is still kept as (nrows, 1, ...)
    # A C-order reshape then gives (nrows*NSBLK, NPOL, NCHAN*NBITS/8)
    tb = sd.data[r0:r1]
    data_2d = np.ascontiguousarray(tb['DATA']).reshape(nrows * nsblk, npol, nchan * nbits // 8)
    ofs = s0 - r0 * nsblk
    n_samp = s1 - s0
    data_packed = data_2d[ofs:ofs + n_samp]            # (n_samp, npol, packed_dim)
    del data_2d, tb
    gc.collect()

    # Frequency axis (use DAT_FREQ from the first row inside the window)
    f_axis = sd.data[r0]['DAT_FREQ'].astype(float)     # (nchan,) MHz
    f_min, f_max = float(f_axis.min()), float(f_axis.max())
    freq_ascending = bool(f_axis[-1] > f_axis[0])
    if not freq_ascending:
        print('[info] frequencies are in descending order; flipping so that the '
              'bottom = low frequency')

    # ---- Unpack ----
    data_unpack = unpack_data_vectorized(data_packed, nbits, nchan, n_samp, npol)
    del data_packed
    gc.collect()
    print(f'unpacked: {data_unpack.shape}  (nsamp, npol, nchan)')

    # ---- Total-intensity synthesis + frequency-axis orientation ----
    data_out = pick_polarization_intensity(data_unpack, npol, pol_order)
    del data_unpack
    gc.collect()
    if not freq_ascending:
        data_out = np.flipud(data_out)
    print(f'data_out (nchan, nsamp): {data_out.shape}')

    # ---- Remove narrowband RFI (bin first, then zap) ----
    # ---- Automatic binning (guard against extreme windows) ----
    data_rfi = data_out.astype(np.float64, copy=True)
    bs = args.bin_samp if args.bin_samp is not None else 4
    bf = args.bin_chn if args.bin_chn is not None else 16
    # Safety net: enlarge bs further if the binned column count is still huge
    if data_rfi.shape[1] // bs > 12000:
        extra = int(np.ceil((data_rfi.shape[1] // bs) / 12000))
        bs *= extra
        print(f'[auto] time bin factor bs auto-increased to {bs} '
              f'(too many columns)')
    dat_plot = bin_data(data_rfi, bs, bf)
    zap_rfi_freq(dat_plot, dat_plot.shape[0], bs, bf)
    print(f'after bin: {dat_plot.shape}  bs={bs} bf={bf}')

    # ---- Plot ----
    fig, ax = plt.subplots()
    vmin = float(np.mean(dat_plot) - 3 * np.std(dat_plot))
    vmax = float(np.mean(dat_plot) + 3 * np.std(dat_plot))
    extent = (s0 * tsamp, s1 * tsamp, f_min, f_max)
    ax.imshow(dat_plot, vmin=vmin, vmax=vmax, aspect='auto', origin='lower',
              cmap=args.cmap, interpolation='none', extent=extent)
    if args.noaxis:
        ax.axis('off')
    else:
        ax.set_xlabel('Time since file start (s)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_xlim(s0 * tsamp, s1 * tsamp)
        ax.set_ylim(f_min, f_max)
        ax.tick_params(labelsize=8)
        xt = np.linspace(s0 * tsamp, s1 * tsamp, 4)
        ax.set_xticks(xt); ax.set_xticklabels([f'{x:.3f}' for x in xt])
        yt = np.linspace(f_min, f_max, 4)
        ax.set_yticks(yt); ax.set_yticklabels([f'{y:.0f}' for y in yt])

    # Title: file name + UTC start/end
    iso_s = Time(mjd0 + s0 * tsamp / 86400.0, format='mjd').isot
    iso_e = Time(mjd0 + s1 * tsamp / 86400.0, format='mjd').isot
    fname = in_file.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    ax.set_title(f'{fname}\nUTC: {iso_s} - {iso_e}', fontsize=9)

    # ---- Output ----
    if args.output_file:
        out_path = args.output_file
    else:
        stem = fname.rsplit('.', 1)[0]
        out_path = f'./{stem}_{t_start:.3f}_{t_end:.3f}.png'
    fig.savefig(out_path, bbox_inches='tight', dpi=args.dpi, pad_inches=0)
    plt.close(fig)
    hdul.close()
    print(f'saved: {out_path}  (total {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
