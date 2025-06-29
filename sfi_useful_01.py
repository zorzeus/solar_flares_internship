# downloading AIA data

from sunpy.net import Fido, attrs as a
import astropy.units as u

# procedure is similar for euv_12s
query_1600 = Fido.search( 
    a.Time("2015-11-04T13:30", "2015-11-04T14:00"),
    a.jsoc.Series("aia.lev1_uv_24s"),
    a.Wavelength(1600 * u.angstrom),
    a.jsoc.Notify("djoletesla98@gmail.com"))

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

