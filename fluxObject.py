from pylab import *
class fluxObject():
    def __init__(self, filename, z, dist, flux, ivar, mask, finalWave):
        self.filename = filename
        self.z = z
        self.dist = dist
        self.flux = flux
        self.ivar = ivar
        self.mask = mask
        self.finalWave = finalWave