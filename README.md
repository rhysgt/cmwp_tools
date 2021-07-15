# cmwp_tools

A collection of tools and scripts which are useful for maipulating XRD data and performing CMWP

- src/xrd_tools.py (library containing usefult general xrd commands)
- src/cmwp_tools.py (library contining useful commands specific to CMWP)
- data/srim.txt (SRIM data for Zr - 1dpa at 60% depth)
- data/ellipticities.txt (Chk0al, alaL and a2aL for given ellipticity - used for dislocation loop type calculation)
- Extract_data.ipynb (Data extraction from a folder of .sol files)
![Example](img/extract_data.png)
- Zr_DESY_2021.ipynb (Integration of data from DESY experiment and production of .bg-spline.dat and .peak-index.dat)
- Zr_batch.ipynb (Production of .bg-spline.dat and .peak-index.dat for prep-integrated data)
- Make instrumental_new.ipynb - UNFINISHED
