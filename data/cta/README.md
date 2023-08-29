# CTA data files

## IRFs

The CTA IRFs can be downloaded from here: [https://www.cta-observatory.org/science/cta-performance/]
For this simulation the southern array is used with and zenith angle of 20° and cuts optimized for long observation times (50h in this case).   
For the `BACKGROUND` HDU the required shape in gammapy version 0.17 is (lon, lat, E) which is the transposed of the recommended GADF shape (E, lat, lon). One can use the following script to adapt the public IRF file to v0.17:

<details><summary>Click to expand</summary>

``` 
from astropy.io import fits
import numpy as np 

filename_v18 = 'cta/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits'  # file with GADF format (E, lat, lon)
filename_v17 = 'cta/irf_file_new.fits'  # out filename with old format (lon, lat, E)

with fits.open(filename_v18) as hdu:
    bkg=hdu["BACKGROUND"]
    bkg.data["BKG"][0] = bkg.data["BKG"][0].T.reshape(60,60,21) # transpose the data structure but keep old shape
    bkg.columns["BKG"].dim='(60,60,21)'  # give the new shape (fits shape is mirrored numpy shape)
    bkg.update()
    hdu.writeto(filename_v17, overwrite=True)  # write as a new file

```
</details>

The adapted file is stored in `/irfs` and is used for the production of the CTA datasets.

## Pseudodata sets

The pseudodata sets can be produced using the `Analysis/Create_CTA_datasets` notebook. The outcome will be stored in the `/pseudodata` folder.

## Plots
Plots from the IRFs showing the sensitivity towards the sources can be produced using the _Plot_CTA_datasets_ notebook.
