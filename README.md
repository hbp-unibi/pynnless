# PyNNLess

![PyNNLess Logo](https://raw.github.com/hbp-sanncs/pynnless/master/docu/logo.png)

## About

PyNNLess is yet another abstraction layer on top of
[PyNN](http://neuralensemble.org/PyNN/). It aims at providing a simple and
stable API for experiments with relatively small spiking neural networks. It
should work with all backends used in the
[Human Brain Project (HBP)](https://www.humanbrainproject.eu/).

Backends include the software simulator [NEST](http://www.nest-simulator.org/)
(versions 2.2 and 2.4), the
[SpiNNaker multicore system](https://github.com/SpiNNakerManchester/)
(NMMC1) developed at Manchester University and the
[HICANN physical modell](https://github.com/electronicvisions/)
(NMPM1) developed at Heidelberg University and its emulation, the ESS.

### Why yet another PyNN abstraction layer?

_PyNNLess_ provides a common API for both _PyNN_ 0.7 and 0.8 and works around the
bugs in the hardware backend bindings. Eventually, at some point in the future
these bugs will be fixed and _PyNNLess_ will be obsolete.

Both network descriptions and recorded results are provided in a JSON-like
object format, making it very easy to use _PyNNLess_ but rendering it impractical
for larger networks and long simulation times.

You might find _PyNNLess_ interesting if you want to simulate fairly small
networks (both network description and recorded results have to fit into main
memory) and run them on multiple backends or with different _PyNN_ versions.

### How to use

Download the most recent version of _PyNNLess_ using the following command:
```bash
git clone https://github.com/hbp-sanncs/pynnless.git
```

_PyNNLess_ depends on _PyNN_ in either version 0.7 or 0.8. Examples on how to
use _PyNNLess_ can be found in the
[`examples`](https://github.com/hbp-sanncs/pynnless/tree/master/examples)
folder. You can simply execute the examples, you do not need to globally install
_PyNNLess_.

If you want to install _PyNNLess_ on your system you can do so using the
following command (execute from the directory into which you have downloaded
_PyNNLess_):
```bash
sudo pip install pynnless
```

It can be uninstalled with the following command:
```bash
sudo pip uninstall pynnless
```

### Further information

* Information on how to use the HBP Neuromorphic Platform:
	[HBP Guidebook](http://electronicvisions.github.io/hbp-sp9-guidebook/)
* PyNN to SpiNNaker Wrapper:
	[sPyNNaker](https://github.com/SpiNNakerManchester/sPyNNaker)

### Contribute

This project has been tailored to the use-cases required in our own work. If
you'd like to expand the functionality please send a pull request on GitHub.
Feel free to open an issue on GitHub if you think you've found a bug.

## Authors

This project has been initiated by Andreas Stöckel in 2015 as part of his Masters Thesis
at Bielefeld University in the [Cognitronics and Sensor Systems Group](http://www.ks.cit-ec.uni-bielefeld.de/) which is
part of the [Human Brain Project, SP 9](https://www.humanbrainproject.eu/neuromorphic-computing-platform).

## License

This project and all its files are licensed under the
[GPL version 3](http://www.gnu.org/licenses/gpl.txt) unless explicitly stated
differently.

The "tomahawk" logo has been adapted from a drawing by [OpenClipart user Firkin](https://openclipart.org/detail/224515/tomahawk).
