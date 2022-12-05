# `hirespipeline`, a Keck/HIRES Data Reduction Pipeline
I'm writing this README text in December 2022. For some reason, despite being 
one of the most productive instruments at the Keck observatory on Mauna Kea, 
the HIRES[^2] instrument does not have a standard data reduction pipeline. 
There are some exoplanet-specific pipelines which aren't really public, a 
limited-use automatic one that doesn't produce science-quality results (and has 
failed on all of my recent data sets) and some very old IDL-based pipelines 
that also no longer seem to work and are no longer supported by their authors. 
Similarly, the ``PypeIt`` package[^1], intended by its designers to be the 
standard spectroscopic data reduction pipeline for Python, does not (yet) have 
HIRES support.

This repository contains my attempt at making an automated pipeline for 
reducing Keck HIRES observations. I built it primarily for a Galilean satellite 
aurora project I work on at Caltech with Katherine de Kleer, Mike Brown and 
Maria Camarca and a collaborator at Boston University Carl Schmidt. I've 
cobbled together many ideas and components from a variety of sources including 
the Mauna Kea Echelle Extraction ``MAKEE``package[^3] and the ``HIRedux`` IDL 
reduction pipeline [^4] written by Jason X. Prochaska.

If all goes well, this pipeline will:
1. Detect each order and its bounds, including those crossing between
   detectors,
2. Reduce the science frames by subtracting the bias, flat-fielding and
   gain-correcting,
3. Automatically calculate a wavelength solution for each extracted order, and
4. Save the result for easy access and/or archiving (as a FITS file appended 
   with the extension `_reduced`).

I don't yet know if this will work without modification for other types of 
HIRES data (I imagine the use of different filters may change how well it 
operates). However, in the scientific spirit, I've made this repository public 
so anyone can take and modify what I've done here.

[^1]: https://pypeit.readthedocs.io/en/release/
[^2]: https://www2.keck.hawaii.edu/inst/hires/
[^3]: https://sites.astro.caltech.edu/~tb/makee/
[^4]: https://www.ucolick.org/~xavier/HIRedux/

## Installation
I've used this code successfully on a Mac running macOS Monterey and Ventura 
through an Anaconda virtual environment running Python 3.10.6. If I were you, I 
would install this in a virtual environment running Python 3.10 (or newer) so 
you don't mess up any of your other projects.

Here are some installation instructions for the average Anaconda user, if 
you're more advanced I'm sure you can figure it out from here.
1. \[Optional/Recommended\] Create a virtual environment (I've named it
   `hires_reduction` in the example):<br>
   `% conda create --name hires_reduction python=3.10`
2. Activate your virtual environment:<br>
    `% conda activate hires_reduction`
3. Install the `hirespipeline` package and its dependencies:<br>
    `% python -m pip install git+https://github.com/zachariahmilby/keck-hires-pipeline.git`

## Usage
This package has a single class with a single public method, so there isn't too
much you have to do as a user. To begin, simply import the class
```
>>> from hirespipeline import HIRESPipeline
```
I've sorted my data into a directory I've named `selected`, within which I've 
placed a number of sub-directories containing the respective FITS files. Though 
you don't have to use the name `selected` for the parent directory, the 
directory structure within must have subdirectories with the following names 
containing the bias, flat, arc and trace files. There can be multiple `science` 
directories containing actual observations, and you can name them whatever you 
want because you have to directly point to them. 
```
selected
|
└───bias
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│   
└───flat
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│   
└───arc
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│
└───trace
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│
└───science
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
```
Finally, to run the pipeline on a science sub-directory, create a pipeline 
object with the name of the observation target, filepath (relative or absolute) 
to the `selected` directory, and the name of the sub-directory containing the 
science data.
```
>>> pipeline = HIRESPipeline(target='Ganymede', 
                             file_directory='/path/to/selected',
                             science_file_directory='science')
>>> pipeline.run()
```
This will save the reduced science data in a `reduced` directory on the same 
level as `selected`. It also includes a few quality-assurance graphics showing
the traces, echelle order bounds, and identified order numbers and wavelength 
bounds so you can see if it worked well or not.

> **NOTE**<br>
> The wavelength solution should be accurate to within one slit width or so. If
> you are looking for a precise wavelength solution, you will want to identify 
> some known features and calculate an offset.
