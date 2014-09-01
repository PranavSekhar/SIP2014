import pyfits
from pylab import *
from astropy.cosmology import WMAP9 as cos
import pdb
import copy
from fluxObject import fluxObject
from scipy.optimize import curve_fit
from astropy.io import ascii
from astropy.table import Table
import asciitable as asc

fileSet = 'alldeep_egs_general_acs_gm20_raindr3_renbin_xray.cat.fits'
numGalaxies = []

fileUse = pyfits.open(fileSet)
tFile = fileUse[1].data
sum = 0
maskName = tFile.field('MASKNAME')
slitName = tFile.field('SLITNAME')
zQ = tFile.field('ZQUALITY')
inc = np.arccos(tFile.field('BA_I')) * 180 / math.pi
z = tFile.field('ZHELIO')
mass = tFile.field('MASS')

tIndex = np.flatnonzero(
    (z > 0.85) & (z < 1.5) & (zQ >= 4) & (maskName >= '5000') & (inc >= 65) & (mass >= 8) & (mass <= 9.5))

filenameB, filenameR, fluxObjects = [], [], []  # blue side files , red side files, list of fluxObjects
for i in range(len(tIndex)):  # reads files
    fileB = ('slit.' + str(maskName[tIndex[i]]) + '.' + str(slitName[tIndex[i]]).zfill(3) + 'B.fits.gz', z[tIndex[i]])
    fileR = ('slit.' + str(maskName[tIndex[i]]) + '.' + str(slitName[tIndex[i]]).zfill(3) + 'R.fits.gz', z[tIndex[i]])
    filenameB.append(fileB)
    filenameR.append(fileR)

pixel_scale = 0.117371  # pixel to arcsecond ratio from DEEP2


def distToKpc(z, cosmo=None):  # returns Kpc/arcsec
    dist = cos.angular_diameter_distance(z)
    return dist.value * 1000


for i in range(len(filenameB)):
    try:
        fileB = pyfits.open(filenameB[i][0])
        fileR = pyfits.open(filenameR[i][0])
        specB = fileB[1].data
        specR = fileR[1].data
        z = filenameB[i][1]
        try:
            tFlux = np.hstack((specB.field('flux')[0], specR.field('flux')[0]))
            tMask = np.hstack((specB.field('mask')[0], specR.field('mask')[0]))
            tIvar = np.hstack((specB.field('ivar')[0], specR.field('ivar')[0]))
            cWave = np.hstack((specB.field('LAMBDA0'), specR.field('LAMBDA0')))
            dWave = np.hstack((specB.field('DLAMBDA')[0], specR.field('DLAMBDA')[0]))

            nb = dWave.shape  # dimensions of wavelength array
            nSpatial = nb[0]  # spatial (arcseconds) dimension
            nWave = nb[1]

            waveLen = np.reshape(np.repeat(cWave, nSpatial, axis=0), (nSpatial, nWave)) 
            WaveLenXY = waveLen + dWave
            restWaveLenXY = WaveLenXY / (1.0 + z)

            tdist = np.arange(0, nSpatial) * pixel_scale * distToKpc(z) * 4.84813681e-6

            filename = 'slit.' + str(maskName[tIndex[i]]) + '.' + str(slitName[tIndex[i]])
            print len(fluxObjects)
            fluxObjects.append(fluxObject(filename, z, tdist, tFlux, tIvar, tMask, restWaveLenXY))
        except ValueError:
            pass
    except IOError as i:
        print i


def gaus(dist, a, x0, sigma):  # gaussian function
    return a * exp(-(dist - x0) ** 2 / (2 * sigma ** 2))


def trimArr(original, x, y):
    result = np.zeros((shape(original)))
    for i in range(len(x)):
        result[x[i]][y[i]] = original[x[i]][y[i]]
    return result


def analyzeFile(fluxObject, colInfo, wavemin, wavemax, edgeEffect):
    obj = fluxObject
    flux, ivar, finalWave, dist = copy.copy(obj.flux), copy.copy(obj.ivar), copy.copy(obj.finalWave), copy.copy(
        obj.dist)
    edgeIndex = round(edgeEffect / 100.0 * len(obj.flux))
    flux = flux[edgeIndex:len(flux) - edgeIndex]
    ivar = ivar[edgeIndex:len(ivar) - edgeIndex]
    finalWave = finalWave[edgeIndex:len(finalWave) - edgeIndex]
    dist = dist[edgeIndex:len(dist) - edgeIndex]

    x, y = np.nonzero((finalWave > wavemin) & (finalWave < wavemax) & (ivar > 0))
    iFlux = trimArr(flux, x, y)  # target fluxes
    tIvar = trimArr(ivar, x, y)  # target ivars
    ttIvar = np.sum(tIvar, axis=1)
    try:
        meanFlux = np.sum((iFlux * tIvar), axis=1) / ttIvar
        #meanFlux /= np.max(meanFlux)
        popt, pcov = curve_fit(gaus, dist, meanFlux, p0=[np.max(meanFlux), meanFlux.argmax(), 5])  # o2 - 100
        curve = gaus(dist, *popt)
        curve /= np.max(curve)
        dist -= dist[curve.argmax()]
        pdb.set_trace()
    except RuntimeError as e:
        dist -= dist[meanFlux.argmax()]
    print 'index:', len(colInfo)
    colInfo.append((meanFlux, ttIvar, dist))


def siftFiles(colInfo, wavemin, wavemax, edgeEffect):  # goes through a set of files
    global fluxObjects
    for i in range(len(fluxObjects)):
        analyzeFile(fluxObjects[i], colInfo, wavemin, wavemax, edgeEffect)


def stackFluxes(colInfo):  # stacks the collapsed fluxes of a set of objects and returns necessary graphing info
    global numGalaxies
    numGalaxies.append(len(colInfo))
    maxLen = 0
    for i in range(len(colInfo)):
        meanFlux = colInfo[i][0]
        if (maxLen < len(meanFlux)):
            maxLen = len(meanFlux)

    x1 = np.linspace(-20, 20, maxLen)
    sumFlux = np.zeros(len(x1))
    sumIvar = np.zeros(len(x1))

    for i in range(len(colInfo)):
        meanFlux, ivar, dist = colInfo[i][0], colInfo[i][1], colInfo[i][2]
        if (not np.isnan(np.min(meanFlux))):
            # pdb.set_trace()
            tIvar = np.interp(x1, dist, ivar)
            flux = np.interp(x1, dist, meanFlux) * tIvar
            sumFlux += flux
            sumIvar += tIvar
    return x1, sumFlux, sumIvar


def plotGraphs(dist, flux, ivar, graphColor, asciiName):
    plt.axhline(0, color='k')
    # plt.plot(stackedFluxes[0], y / np.max(y), graphColor)
    plt.plot(dist, flux, color=graphColor, label=asciiName, linewidth=3)
    plt.xlabel("Distance along the slit (kpc)", fontsize=30)
    plt.ylabel("Normalized flux", fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)


def doProfile(wavemin, wavemax, edgeEffect, graphColor, asciiName):  # generalized profile-graphing function
    # 1d collapsed object information. colInfo[i] = [sum flux of object i, sum ivar of object i, dist]
    colInfo = []
    siftFiles(colInfo, wavemin, wavemax, edgeEffect)
    stack = stackFluxes(colInfo)
    dist, flux, ivar = asciiWrite(stack, asciiName)
    plotGraphs(dist, flux, ivar, graphColor, asciiName)


def asciiWrite(stackedFluxes, asciiName):
    flux = stackedFluxes[1] / stackedFluxes[2]
    flux = flux / np.max(flux)  # normalized meanFlux
    ivar = (stackedFluxes[2] ** 2) / (np.max(flux) ** 2)  # propagated inverse variance

    data = Table([stackedFluxes[0], flux, ivar], names=['dist', 'flux', 'ivar'])
    ascii.write(data, ('values' + asciiName + '.dat'))
    readData = asc.read('values' + asciiName + '.dat', numpy=True)
    dist, flux, ivar = readData['dist'], readData['flux'], readData['ivar']
    return dist, flux, ivar


# doProfile(2800, 2805, 'r', 'MgII Line')
# doProfile(2808, 2813, 'b', 'MgII Continuum')
# doProfile(3724, 3732, 'r', 'OII')
# doProfile(3735, 3740, 'k', 'OII Continuum')
doProfile(2624, 2628, 10, 'r', 'FeII Line')
doProfile(2630, 2635, 10, 'b', 'FeII Continuum')
# doProfile(4859, 4863, 'r', 'HBeta')
# doProfile(5005, 5009, 'c', 'OIII')

plt.text(8, 0.6, 'N = ' + str(max(numGalaxies)), fontsize=30)
plt.legend(prop={'size': 30})

plt.show()