[PyNNLess Logo](https://raw.github.com/hbp-sanncs/pynless/master/docu/logo.png)
# PyNNLess

## About

PyNNLess is yet another abstraction layer ontop of [PyNN](http://neuralensemble.org/PyNN/).
It aims at providing a simple, common API for experiments with relatively small
spiking neural networks that works with all backends used in the Human Brain
Project (HBP).

Backends include the software simulator [NEST](http://www.nest-simulator.org/)
(versions 2.2 and 2.4), the [SpiNNaker multicore hardware](https://github.com/SpiNNakerManchester/)
(NMMC1) developed at Manchester University and the [HICANN physical modell](https://github.com/electronicvisions/)
(NMPM1) developed at Heidelberg University and its emulation, the ESS.

### Why yet another PyNN abstraction layer?

`PyNNLess` provides a common API for both `PyNN` 0.7 and 0.8 and works around the
bugs in the hardware backend bindings. At some point in the future these bugs
will eventually be fixed and `PyNNLess` will be obsolete.

Both network descriptions and recorded results are provided in a JSON-like
object format, making it very easy to use `PyNNLess` but rendering it impractical
for larger networks and long simulation times.

You should use `PyNNLess` instead of `PyNN` if you want to simulate fairly small
networks (both network description and recorded results have to fit into main
memory) and run them on multiple (hardware) backends or with different `PyNN`
versions.

### Further information

* Information on how to use the HBP Neuromorphic Platform: [HBP Guidebook](http://electronicvisions.github.io/hbp-sp9-guidebook/)
* PyNN to SpiNNaker Wrapper: [sPyNNaker](https://github.com/SpiNNakerManchester/sPyNNaker)

## Authors

This project has been initiated by Andreas Stöckel in 2015 as part of his Masters Thesis
at Bielefeld University in the [Cognitronics and Sensor Systems Group](http://www.ks.cit-ec.uni-bielefeld.de/) which is
part of the [Human Brain Project, SP 9](https://www.humanbrainproject.eu/neuromorphic-computing-platform).

## License

This project and all its files are licensed under the
[GPL version 3](http://www.gnu.org/licenses/gpl.txt) unless explicitly stated
differently.

The "tomahawk" logo has been adapted from a drawing by [OpenClipart user Firkin](https://openclipart.org/detail/224515/tomahawk).
