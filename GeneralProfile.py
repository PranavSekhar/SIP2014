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
from numpy import random

fileSet = 'alldeep_egs_general_acs_gm20_raindr3_renbin_xray.cat.fits'
numGalaxies = []

fileUse = pyfits.open(fileSet)
tFile = fileUse[1].data
sum = 0
maskName = tFile.field('MASKNAME')
slitName = tFile.field('SLITNAME')
objno = tFile.field('OBJNO')
zQ = tFile.field('ZQUALITY')
inc = np.arccos(tFile.field('BA_I')) * 180 / math.pi
z = tFile.field('ZHELIO')
mass = tFile.field('MASS')

badGals = [13009695, 12023846, 12016585, 11050883, 13017342, 12009274, 13040223, 13026702, 13025944, 13017369, 12008266,
           11034403, 12004801, 12016414, 13019291, 13033219, 13057589, 13035546, 13035546, 12029435, 12032751, 1300904,
           13019906, 13033219, 13043600, 13057589, 11019338, 14019427, 13064938, 13064938, 11038982, 13056261, 12028185,
           12032038]
tIndex = np.flatnonzero((z > 0.7) & (z < 1.5) & (zQ >= 4) & (maskName >= '5000')
                        & ([x not in badGals for x in objno]))

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

            waveLen = np.reshape(np.repeat(cWave, nSpatial, axis=0), (nSpatial, nWave))  # rename to cwave0
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
        popt, pcov = curve_fit(gaus, dist, meanFlux, p0=[np.max(meanFlux), meanFlux.argmax(), 5])  # o2 - 100
        curve = gaus(dist, *popt)
        dist -= dist[curve.argmax()]
        # pdb.set_trace()
    except RuntimeError as e:
        dist -= dist[meanFlux.argmax()]
    print 'index:', len(colInfo)
    colInfo.append((meanFlux, ttIvar, dist))


def siftFiles(colInfo, wavemin, wavemax, edgeEffect):  # goes through a set of files
    global fluxObjects
    for i in range(len(fluxObjects)):
        analyzeFile(fluxObjects[i], colInfo, wavemin, wavemax, edgeEffect)


def stackFluxes(colInfo, boot):
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
    listFlux = []
    for i in range(len(colInfo)):
        meanFlux, ivar, dist = colInfo[i][0], colInfo[i][1], colInfo[i][2]
        if (not np.isnan(np.min(meanFlux))):
            # pdb.set_trace()
            tIvar = np.interp(x1, dist, ivar)
            flux = np.interp(x1, dist, meanFlux) * tIvar
            sumFlux += flux
            sumIvar += tIvar
            listFlux.append(flux)
    listFlux = np.array(listFlux)
    if (boot == True):
        nsims = 1000
        bootProfiles = []
        for i in range(nsims):
            randIndex = np.random.random_integers(0, high=len(listFlux) - 1, size=len(listFlux))
            sum = np.sum(listFlux[randIndex], axis=0)
            bootProfiles.append(sum)
        err = np.std(bootProfiles, axis=0)
        sumIvar = 1.0 / (err ** 2)
    return x1, sumFlux, sumIvar


def plotGraphs(dist, flux, ivar, graphColor, asciiName):
    plt.figure(1)
    plt.subplot(211)
    plt.axhline(0, color='k')

    plt.errorbar(dist, flux, yerr=np.sqrt(1.0 / ivar), marker='o', ls='None', color=graphColor, label=asciiName)

    # plt.xlabel("Distance along the slit (kpc)", fontsize=20)
    plt.ylabel("Normalized flux", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.text(8, 0.6, 'N = ' + str(max(numGalaxies)), fontsize=15)
    plt.legend(prop={'size': 15})


def doProfile(wavemin, wavemax, edgeEffect, graphColor, asciiName, boot):  # generalized profile-graphing function
    # 1d collapsed object information. colInfo[i] = [sum flux of object i, sum ivar of object i, dist]
    colInfo = []
    siftFiles(colInfo, wavemin, wavemax, edgeEffect)
    stack = stackFluxes(colInfo, boot)
    dist, flux, ivar = asciiWrite(stack, asciiName)
    plotGraphs(dist, flux, ivar, graphColor, asciiName)


def asciiWrite(stackedFluxes, asciiName):
    max = np.max(stackedFluxes[1])
    flux = stackedFluxes[1] / max  # normalized meanFlux
    ivar = (stackedFluxes[2]) * (max ** 2)  # propagated inverse variance

    data = Table([stackedFluxes[0], flux, ivar], names=['dist', 'flux', 'ivar'])
    ascii.write(data, ('values' + asciiName + '.dat'))
    readData = asc.read('values' + asciiName + '.dat', numpy=True)
    dist, flux, ivar = readData['dist'], readData['flux'], readData['ivar']
    return dist, flux, ivar


def plotLC(line, continuum):
    lData = asc.read('values' + line + '.dat', numpy=True)
    cData = asc.read('values' + continuum + '.dat', numpy=True)
    lDist, lFlux, lIvar = lData['dist'], lData['flux'], lData['ivar']
    cDist, cFlux, cIvar = cData['dist'], cData['flux'], cData['ivar']
    if (len(lDist) > len(cDist)):
        dist = lDist
        flux = lFlux - np.interp(dist, cDist, cFlux)
        err = np.sqrt(1.0 / lIvar) + np.sqrt(1.0 / np.interp(dist, cDist, cIvar))
    else:
        dist = cDist
        flux = np.interp(dist, lDist, lFlux) - cFlux
        err = np.sqrt(1.0 / np.interp(dist, lDist, lIvar)) + np.sqrt(1.0 / cIvar)
    plt.subplot(212)
    plt.xlabel("Distance along the slit (kpc)", fontsize=20)
    plt.ylabel("Flux Difference", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.axhline(0, color='k')
    plt.errorbar(dist, flux, err, marker='o')




doProfile(2792, 2806, 0, 'r', 'MgII Line', boot=True) # the other magnesium
doProfile(2808, 2811, 0, 'b', 'MgII Continuum', boot=True)

# doProfile(3724, 3732, 'r', 'OII')
# doProfile(3735, 3740, 'k', 'OII Continuum')

# doProfile(2623, 2627, 0, 'r', 'FeII Line', boot=True)
# doProfile(2632, 2636, 0, 'b', 'FeII Continuum', boot=True)

# doProfile(4859, 4863, 'r', 'HBeta')
# doProfile(5005, 5009, 'c', 'OIII')
plotLC('MgII Line', 'MgII Continuum')

plt.show()