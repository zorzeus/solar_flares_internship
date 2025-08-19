sci_query = Fido.search(a.Time("2024-10-15 18:00", "2024-10-15 19:00"), a.Instrument.stix, a.stix.DataType.sci)
cpd_i = []
for i, result in enumerate(sci_query[0]):
    if 'cpd' in result['DataProduct']:
        cpd_i.append(i)

relevant_files = [i for i in cpd_i if i in [1, 2, 3]]
sci_files = Fido.fetch(sci_query[0][relevant_files])
cpd_products = []
for file in sci_files:
    if 'cpd' in str(file):
        cpd_products.append(Product(file))
cpd_flare = cpd_products[2]

def time_correction(stix_times, correction_seconds=281):
    if hasattr(stix_times, 'datetime'):  
        stix_times = stix_times.datetime
    return pd.DatetimeIndex(stix_times) + pd.Timedelta(seconds=correction_seconds)

def stix_energy_bands(lcs, time_stix):
    energy_bands = {
        '4-10 keV': np.sum(lcs[:, 1:7], axis=1),     # channels 1-6: ~4-10 keV
        '10-15 keV': np.sum(lcs[:, 7:12], axis=1),   # channels 7-11: ~10-15 keV  
        '15-22 keV': np.sum(lcs[:, 12:16], axis=1),  # channels 12-15: ~15-22 keV
        '22-45 keV': np.sum(lcs[:, 16:22], axis=1),  # channels 16-21: ~22-45 keV
        '45-100 keV': np.sum(lcs[:, 22:29], axis=1)  # channels 22-28: ~45-100 keV
    }
    
    stix_lightcurves = {}
    for band_name, counts in energy_bands.items():
        stix_lightcurves[band_name] = pd.Series(counts, index=time_stix, name=band_name)
    return stix_lightcurves

def lcs_from_cpd(cpd_product):
    time_stix = cpd_product.data['time']
    counts_raw = cpd_product.data['counts']  
    lcs = np.sum(counts_raw, axis=(2, 3))
    time_stix_corrected = time_correction(time_stix, 281)
    # pre_flare_mask = time_stix_corrected < pd.Timestamp('2024-10-15 18:30:00')
    # background = np.mean(lcs[pre_flare_mask], axis=0, keepdims=True)
    # lcs_clean = lcs - background
    
    return lcs, time_stix_corrected  
lcs, time_stix_corrected = lcs_from_cpd(cpd_flare)
stix_lcs = stix_energy_bands(lcs, time_stix_corrected)

plt.figure(figsize=(12,5.5))
for band_name, lc in stix_lcs.items():
    plt.plot(lc.index, lc.values, label=band_name, linewidth=1)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
plt.xticks(rotation=45)
plt.xlabel(r'Time $(2024/10/15)$ UT')
plt.ylabel('Total counts')
plt.yscale('log')
plt.title('Solar Orbiter/STIX CPDs [time correction applied]')
plt.legend()
plt.tight_layout()
plt.show()