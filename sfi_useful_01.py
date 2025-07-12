# some useful stuff, related to internship

from sunpy.net import Fido, attrs as a
import astropy.units as u

query_1600 = Fido.search( 
    a.Time("2015-11-04T13:30", "2015-11-04T14:00"),
    a.jsoc.Series("aia.lev1_uv_24s"),
    a.Wavelength(1600 * u.angstrom),
    a.jsoc.Notify("djoletesla98@gmail.com")) # procedure is similar for euv_12s

print(query_1600)
files_1600 = Fido.fetch(query_1600)
print(files_1600)

# _____________________________________________________________________________
# _____________________________________________________________________________

# handling the data and plotting lightcurve for AIA

p = '/home/zorzeus/sunpy/data'
files171 = sorted(glob.glob(f'{p}/aia.lev1_euv_12s*.image_lev1.fits'))

times171 = []
roi_means = []

first_map = Map(files171[0])
bottom_left = SkyCoord(-200*u.arcsec,   0*u.arcsec, frame=first_map.coordinate_frame)
top_right   = SkyCoord( 200*u.arcsec, 400*u.arcsec, frame=first_map.coordinate_frame)

for fn in files171:
    m = Map(fn)
    times171.append(pd.Timestamp(m.date.iso))
    sub = m.submap(bottom_left=bottom_left, top_right=top_right)
    data = sub.data
    if np.isnan(data).all():
        roi_means.append(np.nan)
    else:
        roi_means.append(np.nanmean(data))
    del m  

lc171 = pd.Series(roi_means, index=pd.DatetimeIndex(times171))
lc171 = lc171.interpolate(method='time').dropna()

norm171 = (lc171 - lc171.min()) / (lc171.max() - lc171.min())
norm171 = norm171.truncate("2015-11-04T13:30", "2015-11-04T14:00")
norm171 = norm171.resample('12s').mean().interpolate(method='time')

plt.figure(figsize=(10, 4))
plt.plot(norm171.index, norm171, c='navy', lw=2, label=r'AIA $171\,\AA$')
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
plt.xlabel('Time (2015-11-04) UT')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.tight_layout()
plt.show()

# _____________________________________________________________________________
# _____________________________________________________________________________

# wavelet with debugging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline
from scipy.ndimage import label
from matplotlib.ticker import LogLocator
from scipy.signal import savgol_filter
from waveletFunctions import wavelet, wave_signif

def analyze_morlet_wavelet(
    series: pd.Series,
    dj: float = 0.25,
    s0_factor: float = 2.0,
    pad: int = 1,
    k0: float = 6.0,
    samples_between_knots: int = 28,
    detrend: bool = True,
    detrend_method: str = "spline"):

    if isinstance(series, pd.DataFrame):
        if series.shape[1] > 1:
            raise ValueError("Input must be single-column DataFrame or Series.")
        y = series.iloc[:,0].dropna().sort_index()
    else:
        y = series.dropna().sort_index()
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("Index must be pandas DateTimeIndex.")

    t = (y.index - y.index[0]).total_seconds().values
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("Non-positive time step detected.")
    N = len(y)
    s0 = s0_factor * dt
    J1 = int(np.log2(N * dt / s0) / dj)
    print(f"DEBUG: N={N}, dt={dt:.3f}s, s0={s0:.3f}s, dj={dj}, J1={J1}")

    if detrend:
        if detrend_method == "spline":
            print(f"DEBUG: Spline detrending with {samples_between_knots} samples between knots.")
            knot_idxs = np.arange(samples_between_knots, N, samples_between_knots)
            knot_times = t[knot_idxs]
            if len(knot_times) < 2:
                raise ValueError("Insufficient knot points for spline.")
            spline = LSQUnivariateSpline(t, y.values, knot_times, k=3)
            detrended = y.values - spline(t)
        elif detrend_method == "savgol":
            print("DEBUG: Savitzky-Golay detrending with ~300s window.")
            win = int(500/ dt)
            if win % 2 == 0:
                win += 1
            baseline = savgol_filter(y.values, window_length=win, polyorder=3, mode='interp')
            detrended = y.values - baseline
        else:
            raise ValueError("Unknown detrending method: choose 'spline' or 'savgol'.")
    else:
        print("DEBUG: Skipping detrending.")
        detrended = y.values.copy()
    norm_det = detrended / np.max(np.abs(detrended))

    wave, period, scale, coi = wavelet(detrended, dt, pad=pad, dj=dj, s0=s0,
                                        J1=J1, mother='MORLET', param=k0)
    power = np.abs(wave)**2
    var = np.var(detrended, ddof=1)
    lag1 = np.corrcoef(detrended[:-1], detrended[1:])[0,1]
    signif_local = wave_signif(var, dt, scale, sigtest=0, lag1=lag1, mother='MORLET')
    dof = np.maximum(1, N - scale)
    signif_global = wave_signif(var, dt, scale, sigtest=1, lag1=lag1, dof=dof, mother='MORLET')

    global_ws = power.mean(axis=1)
    idx137 = np.argmin(np.abs(period - 137))
    print(f"DEBUG: At 137s: ws={global_ws[idx137]:.2e}, sig={signif_global[idx137]:.2e}, ratio={global_ws[idx137]/signif_global[idx137]:.2f}")
    mask = global_ws > signif_global
    labeled_arr, nregions = label(mask)
    print(f"DEBUG: Found {nregions} significant region(s)")
    regions = []
    for i in range(1, nregions+1):
        idxs = np.where(labeled_arr==i)[0]
        ws_i = global_ws[idxs]
        per_i = period[idxs]
        pk = idxs[np.argmax(ws_i)]
        regions.append({
            'peak': period[pk],
            'left': per_i.min(),
            'right': per_i.max(),
            'power': ws_i.max(),
            'ratio': ws_i.max()/signif_global[pk]})
        r = regions[-1]
        print(f"  Region {i}: peak={r['peak']:.1f}s, range=[{r['left']:.1f},{r['right']:.1f}]s, power={r['power']:.2e}, ratio={r['ratio']:.2f}")
    main = max(regions, key=lambda r: r['ratio'])
    P0 = main['peak']; err_low = P0-main['left']; err_high = main['right']-P0
    print(f"DEBUG: Selected P0={P0:.1f}s (-{err_low:.1f}/+{err_high:.1f}s)")

    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2,4, height_ratios=[2,3], width_ratios=[1,1,1,1], hspace=0.1, wspace=0.2)
   
    ax_ts = fig.add_subplot(gs[0,0:3])
    ax_ts.plot(t, norm_det, 'k', lw=1)
    ax_ts.set_xlim(t.min(), t.max())
    ax_ts.set_ylim(-1,1) if detrend else ax_ts.set_ylim(0,1)
    ax_ts.set_ylabel("Normalized Intensity")
    ax_ts.set_title("Morlet Wavelet Analysis")
    ax_ts.xaxis.set_visible(False)
    
    ax_box = fig.add_subplot(gs[0,3]); ax_box.axis('off')
    ax_box.text(0.5, 0.5, rf"$\rm{{P}} = {P0:.1f} ^{{+{err_high:.1f}}}_{{-{err_low:.1f}}}\,\rm{{s}}$",
    ha='center', va='center', transform=ax_box.transAxes, fontsize=12,
    bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='indianred', alpha=0.3, linewidth=2))
    
    ax_wps = fig.add_subplot(gs[1,0:3]);
    ax_wps.set_facecolor('black')
    T,Pm = np.meshgrid(t,period)
    ax_wps.contourf(T, Pm, power, levels=10, cmap='cividis', extend='both')
    ax_wps.contour(T, Pm, power/signif_local[:,None], levels=[1], colors='orange', linewidths=1)
    ax_wps.plot(t, coi, 'w-', lw=1.5)
    ax_wps.fill_between(t, coi, 1000,facecolor='none',edgecolor='white',alpha=0.3)
    ax_wps.axhline(P0, color='white', ls='--',lw=1)
    ax_wps.set_xlabel("Time (s)")
    ax_wps.set_ylabel("Period (s)")
    ax_wps.set_yscale('log')
    ax_wps.set_ylim(10,1000); ax_wps.set_yticks([10,100,1000])
    
    ax_gws = fig.add_subplot(gs[1,3],sharey=ax_wps)
    norm_ws = global_ws/np.nanmax(global_ws)
    norm_sig = signif_global/np.nanmax(global_ws)
    ax_gws.plot(norm_ws,period,'k')
    ax_gws.plot(norm_sig,period,'k--')
    ax_gws.axhline(P0,color='red',ls='--',lw=1)
    ax_gws.set_xscale('log')
    ax_gws.set_xlim(0.01,1)
    # ax_gws.xaxis.set_major_locator(LogLocator(base=10,numticks=5))
    # ax_gws.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9],numticks=12))
    # ax_gws.set_xticks([1e-4,1e-3,1e-2,1e-1,1]); ax_gws.set_xticklabels(['0.0001','0.001','0.01','0.1','1'])
    ax_gws.set_xlabel("Normalized Power"); ax_gws.yaxis.set_visible(False)
    # ax_gws.legend(loc='lower left')
    return P0, err_low, err_high, figs