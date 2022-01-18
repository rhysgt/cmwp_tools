import os, re, glob
import numpy as np
import fabio
from scipy import optimize
from lmfit import minimize, Parameters, fit_report
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline


def load_tifs(filename):
    """ Load tiff file using fabio.
    
    Params
    -------
    filename: str
        Path to tiff file.
    
    """
    if isinstance(filename, list):  # a list of filenames is given: return the average (mean) image as a numpy array
        data = []
        for f in filename:
            try:
                data.append(fabio.open(f).data)
            except:
                print('Warning: file %s skipped (error while opening image with fabio)' % f)
        assert len(data) > 0, 'Error in load_tifs(): no data found!'
        data = np.mean(np.array(data), axis=0)
    else:  # just one image
        assert os.path.isfile(filename), 'Error in load_tifs(): filename does not exist!'
        data = fabio.open(filename).data
    return data


def integrate_join(ai1, ai2, file1, file2, cutoffs, limits, azimuth=None):
    """ Integrate images from two detectors and join together.
    
    Params
    ------
    ai1, ai2:
        pyFAI azimuthal integrator objects.
    file1, file2: 
        path to tiff files.
    cutoffs: float or [float, float]
        two theta values where data will be joined.
    limits: [float, float]
        total two theta range [min, max]  
    """

    if azimuth == None:
        int1 = ai1.integrate1d(file1, 10000, correctSolidAngle=True, polarization_factor=0.99, method='full_csr',
                               unit='2th_deg')
        int2 = ai2.integrate1d(file2, 10000, correctSolidAngle=True, polarization_factor=0.99, method='full_csr',
                               unit='2th_deg')
    else:
        int1 = ai1.integrate1d(file1, 10000, correctSolidAngle=True, azimuth_range=azimuth, polarization_factor=0.99,
                               method='csr', unit='2th_deg')
        int2 = ai2.integrate1d(file2, 10000, correctSolidAngle=True, azimuth_range=azimuth, polarization_factor=0.99,
                               method='csr', unit='2th_deg')

    # Find limit index
    limits[0] = int(np.abs(int1[0] - limits[0]).argmin())
    limits[1] = int(np.abs(int2[0] - limits[1]).argmin())

    # Find cutoff index
    cutoff_index = [0, 0]
    if isinstance(cutoffs, float):
        cutoff_index[0] = int(np.abs(int1[0] - cutoffs).argmin())
        cutoff_index[1] = int(np.abs(int2[0] - cutoffs).argmin()) + 1
    else:
        cutoff_index[0] = int(np.abs(int1[0] - cutoffs[0]).argmin())
        cutoff_index[1] = int(np.abs(int2[0] - cutoffs[1]).argmin()) + 1

    # Find shift between detectors where they join
    shift = np.mean(int2[1][cutoff_index[1]:cutoff_index[1] + 50] - int1[1][cutoff_index[0] - 50:cutoff_index[0]])

    intnew = np.array([np.concatenate((int1[0][limits[0]:cutoff_index[0]], int2[0][cutoff_index[1]:limits[1]])),
                       np.concatenate(
                           (int1[1][limits[0]:cutoff_index[0]], int2[1][cutoff_index[1]:limits[1]] - shift))])

    # Add offset to make numbers positive
    intnew[1] -= np.min(intnew[1])
    intnew[1] += 10

    return intnew


def calculateObjective(params, hkls, poss, wavelength):
    """
    This is the objective function based on the Hull-Davey equation 
    to calculate lattice paramaters.
    
    """
    vals = params.valuesdict()
    a = vals['a']
    c = vals['c']

    objective = []
    for hkl, pos in zip(hkls, poss):
        h = int(hkl[0]);
        k = int(hkl[1]);
        l = int(hkl[2]);
        K = 2 * np.sin(np.deg2rad(pos / 2)) / wavelength
        logd2 = 2 * np.log(1 / K)
        eq = 2 * np.log(a) - np.log((4 / 3) * (h ** 2 + h * k + k ** 2) + l ** 2 / (c / a) ** 2)
        weight = np.tan(np.deg2rad(pos / 2))
        diff = (logd2 - eq) ** 2 * weight ** 2

        objective.append(diff)

    return objective


def calculateLatticeParams(hkls, poss, wavelength, printResult=False, plotResult=False):
    """
    Uses minimisation of an objective function `calculateObjective`
    based on the Hull-Davey formula to calculate lattice paramaters.
    
    Params
    ------
    hkls: list(str)
        HKLs.
    poss: list(float)
        Peak positions.
    wavelength: float
        Wavelength in nanometer.
    printResult: bool
        Prints result if true.
    
    """
    fit_params = Parameters()
    fit_params.add('a', value=0.3232, min=0.31, max=0.34)
    fit_params.add('c', value=0.5147, min=0.50, max=0.53)

    res = minimize(calculateObjective, params=fit_params, args=(hkls, poss), kws={'wavelength': wavelength})

    if printResult:
        print(fit_report(res))

    if plotResult == True:
        a = res.params['a'].value
        c = res.params['c'].value
        fig, ax = plt.subplots(figsize=(10, 5))
        for hkl, pos in zip(hkls, poss):
            h = int(hkl[0]);
            k = int(hkl[1]);
            l = int(hkl[2]);
            K = 2 * np.sin(np.deg2rad(pos / 2)) / wavelength
            logd2 = 2 * np.log(1 / K)
            eq = 2 * np.log(a) - np.log((4 / 3) * (h ** 2 + h * k + k ** 2) + l ** 2 / (c / a) ** 2)
            weight = np.tan(np.deg2rad(pos / 2))
            diff = (logd2 - eq) ** 2 * weight ** 2
            ax.scatter(logd2, eq, c='k')
        plt.xlabel('2log(1/K)')
        plt.ylabel('2*np.log(a)-np.log((4/3)*(h**2+h*k+k**2)+l**2/(c/a)**2)')

    return res


class dislocationTypeCalc:
    """
    Class used for calculation of dislocation type from given `a1` and `a2` values for a hcp crystal.
    
    Example
    --------
    import cmwp_tools
    cmwp_tools.dislocationTypeCalc(disloctype='loop', ellipticity=1).calcRho(a1-0.5623, a2=-0.1256, rho=19.6, printResult=True)
    cmwp_tools.dislocationTypeCalc(disloctype='def').calcRhoError(a1=(0.5623, 0.1, 0.2), a2=(-0.1256, 0.12, 0.5), rho=19.6, printResult=True)
    cmwp_tools.dislocationTypeCalc(disloctype='both').calcRhoError(a1=(0.5623, 0.1, 0.2), a2=(-0.1256, 0.12, 0.5), rho=19.6, printResult=True)
    
    a1 and a2 format (VALUE, NEGATIVE ERROR, POSITIVE ERROR)
    
    """

    def __init__(self, disloctype, ellipticity=None):

        if disloctype == 'def' or 'loop' or 'both':
            self.disloctype = disloctype
        else:
            raise ValueError('Incrorrect disloctype - should be def or loop or both')

        self.a_param = 0.3232;
        self.c_param = 0.5147;
        self.b = 0.3232;
        self.Chk0 = 0.25;
        self.hkls = ['100', '002', '101', '102', '110', '200', '112', '201', '004', '202',
                     '104', '203', '210', '211', '114', '105', '204', '300', '213', '302', '006']

        if self.disloctype == 'def' or self.disloctype == 'both':
            self.Chk0a = 0.2368;
            self.a1a = -0.0436;
            self.a2a = -0.3011;
            self.ba = 0.3232;
            self.Chk0c = 0.045;
            self.a1c = 5.8513;
            self.a2c = 3.1882;
            self.bc = 0.5140;
            self.Chk0ca = 0.0897;
            self.a1ca = 2.4520;
            self.a2ca = 0.7474;
            self.bca = 0.6078;

        if self.disloctype == 'loop' or self.disloctype == 'both':

            self.Chk0cL = 0.0815;
            self.a1cL = 2.6411;
            self.a2cL = 1.0102;
            self.baL = 0.3232;
            self.bcL = 0.2790;

            if ellipticity == None:
                raise ValueError('Please specify ellipticity')
            else:
                self.ellipticity = ellipticity
                self.calcEllipticity()

    def calcH1H2(self, hkl):

        h = int(hkl[0]);
        k = int(hkl[1]);
        l = int(hkl[2])
        h1_squared = (h ** 2 + k ** 2 + (h + k) ** 2) * l ** 2 / (
                    h ** 2 + k ** 2 + (h + k) ** 2 + 3 / 2 * (self.a_param / self.c_param) ** 2 * l ** 2) ** 2
        h2_squared = l ** 4 / (
                    h ** 2 + k ** 2 + (h + k) ** 2 + 3 / 2 * (self.a_param / self.c_param) ** 2 * l ** 2) ** 2
        return h1_squared, h2_squared

    def calcEpsSquaredExp(self, hkl, a1, a2, rho):

        h = int(hkl[0]);
        k = int(hkl[1]);
        l = int(hkl[2])
        h1_squared, h2_squared = self.calcH1H2(hkl)
        C = self.Chk0 * (1 + (a1 * h1_squared) + (a2 * h2_squared))
        EpsCalcSquared = rho * C * self.b ** 2
        return EpsCalcSquared

    def calcEpsSquaredThe(self, hkl, a_am=None, c_am=None, ca_am=None, RoaL=None, RocL=None):

        h = int(hkl[0]);
        k = int(hkl[1]);
        l = int(hkl[2])
        h1_squared, h2_squared = self.calcH1H2(hkl)

        EpsSquaredThe = 0

        if self.disloctype == 'def' or self.disloctype == 'both':
            Ca = self.Chk0a * (1 + self.a1a * h1_squared + self.a2a * h2_squared)
            Cc = self.Chk0c * (1 + self.a1c * h1_squared + self.a2c * h2_squared)
            Cca = self.Chk0ca * (1 + self.a1ca * h1_squared + self.a2ca * h2_squared)

            EpsSquaredThe += a_am * Ca * self.ba ** 2 + c_am * Cc * self.bc ** 2 + ca_am * Cca * self.bca ** 2

        if self.disloctype == 'loop' or self.disloctype == 'both':
            CaLoop = self.Chk0al * (1 + self.a1aL * h1_squared + self.a2aL * h2_squared)
            CcLoop = self.Chk0cL * (1 + self.a1cL * h1_squared + self.a2cL * h2_squared)

            EpsSquaredThe += RoaL * CaLoop * self.baL ** 2 + RocL * CcLoop * self.bcL ** 2

        return EpsSquaredThe

    def minimisationfunct(self, params, a1, a2, rho):

        if self.disloctype == 'def':
            a_am, c_am, ca_am = params
            totlist = []
            for hkl in self.hkls:
                DiffSquared = (
                            self.calcEpsSquaredExp(hkl, a1, a2, rho) - self.calcEpsSquaredThe(hkl, a_am=a_am, c_am=c_am,
                                                                                              ca_am=ca_am))
                totlist.append(DiffSquared)

        if self.disloctype == 'loop':
            RoaL, RocL = params
            totlist = []
            for hkl in self.hkls:
                DiffSquared = (self.calcEpsSquaredExp(hkl, a1, a2, rho) - self.calcEpsSquaredThe(hkl, RoaL=RoaL,
                                                                                                 RocL=RocL))
                totlist.append(DiffSquared)

        if self.disloctype == 'both':
            a_am, c_am, ca_am, RoaL, RocL = params
            totlist = []
            for hkl in self.hkls:
                DiffSquared = (
                            self.calcEpsSquaredExp(hkl, a1, a2, rho) - self.calcEpsSquaredThe(hkl, a_am=a_am, c_am=c_am,
                                                                                              ca_am=ca_am, RoaL=RoaL,
                                                                                              RocL=RocL))
                totlist.append(DiffSquared)

        return np.linalg.norm(totlist)

    def calcRho(self, a1, a2, rho, printResult=False):

        if printResult is True:
            print('a1 = {0:.4f}\na2 = {1:.4f}\nrho = {2:.4f}\n'.format(a1, a2, rho))

        if self.disloctype == 'def':
            res = optimize.minimize(self.minimisationfunct, x0=[rho / 3, rho / 3, rho / 3],
                                    args=(a1, a2, rho), bounds=((0.001, 200), (0.001, 200), (0.001, 200)),
                                    method='SLSQP', options={'ftol': 1e-9, 'eps': 1e-9})

            a = res.x[0] / np.sum(res.x);
            c = res.x[1] / np.sum(res.x);
            ac = res.x[2] / np.sum(res.x)
            rho_new = np.sum(res.x)

            if printResult is True:
                print('<a> proportion: {0:.4f}'.format(a))
                print('<c> proportion: {0:.4f}'.format(c))
                print('<c+a> proportion: {0:.4f}'.format(ac))
                print('New rho: {0:.3f} 10^14 m^-2'.format(rho_new))

            return a, c, ac, rho_new

        if self.disloctype == 'loop':
            res = optimize.minimize(self.minimisationfunct, x0=[rho / 2, rho / 2],
                                    args=(a1, a2, rho), bounds=((0.001, 200), (0.001, 200)),
                                    method='SLSQP', options={'ftol': 1e-9, 'eps': 1e-9})

            ac = res.x[0] / np.sum(res.x)
            rho_new = np.sum(res.x)

            if printResult is True:
                print('ellipticity = {0}\n'.format(self.ellipticity))
                print('<a> loop proportion: {0:.4f}'.format(ac))
                print('New rho: {0:.4f} 10^14 m^-2'.format(rho_new))

            return ac, rho_new

        if self.disloctype == 'both':
            res = optimize.minimize(self.minimisationfunct, x0=[rho / 5, rho / 5, rho / 5, rho / 5, rho / 5],
                                    args=(a1, a2, rho),
                                    bounds=((0.001, 200), (0.001, 200), (0.001, 200), (0.001, 200), (0.001, 200)),
                                    method='SLSQP', options={'ftol': 1e-9, 'eps': 1e-9})

            a = res.x[0] / np.sum(res.x);
            c = res.x[1] / np.sum(res.x);
            ac = res.x[2] / np.sum(res.x);
            a_loop = res.x[3] / np.sum(res.x);
            c_loop = res.x[4] / np.sum(res.x);
            rho_new = np.sum(res.x)

            if printResult is True:
                print('ellipticity = {0}\n'.format(self.ellipticity))
                print('<a> proportion: {0:.4f}'.format(a))
                print('<c> proportion: {0:.4f}'.format(c))
                print('<c+a> proportion: {0:.4f}'.format(ac))
                print('<a> loop proportion: {0:.4f}'.format(a_loop))
                print('<c> loop proportion: {0:.4f}'.format(c_loop))
                print('New rho: {0:.4f} 10^14 m^-2'.format(rho_new))

            return a, c, ac, a_loop, c_loop, rho_new

    def calcRhoError(self, a1, a2, rho, printResult):

        a1, a1_neg, a1_pos = a1
        a2, a2_neg, a2_pos = a2

        combilist = [[a1 - a1_neg, a2], [a1 + a1_pos, a2], [a1, a2 + a2_pos], [a1, a2 - a2_neg],
                     [a1 - a1_neg / 2, a2 - a2_neg / 2], [a1 + a1_pos / 2, a2 + a2_pos / 2],
                     [a1 - a1_neg / 2, a2 + a2_pos / 2], [a1 + a1_pos / 2, a2 - a2_neg / 2]]

        if self.disloctype == 'def':

            a, c, ac, rho_new = self.calcRho(a1, a2, rho, printResult=printResult)

            error_a_list = [];
            error_c_list = [];
            error_ac_list = [];

            for item in combilist:
                error_a, error_c, error_ca, error_rho = self.calcRho(item[0], item[1], rho, printResult=False)
                error_a_list.append(error_a)
                error_c_list.append(error_c)
                error_ac_list.append(error_ca)

            error_a_neg = a - np.min(error_a_list);
            error_a_pos = np.abs(np.max(error_a_list) - a);
            error_c_neg = c - np.min(error_c_list);
            error_c_pos = np.abs(np.max(error_c_list) - c);
            error_ac_neg = ac - np.min(error_ac_list);
            error_ac_pos = np.abs(np.max(error_ac_list) - ac);

            if printResult == True:
                print('<a> proportion: {0:.4f}'.format(a))
                print('-: {0:.3f}'.format(error_a_neg))
                print('+: {0:.3f}'.format(error_a_pos))
                print('<c> proportion: {0:.4f}'.format(c))
                print('-: {0:.3f}'.format(error_c_neg))
                print('+: {0:.3f}'.format(error_c_pos))
                print('<c+a> proportion: {0:.4f}'.format(ac))
                print('-: {0:.3f}'.format(error_ac_neg))
                print('+: {0:.3f}'.format(error_ac_pos))
                print('New rho: {0:.3f} 10^14 m^-2'.format(rho_new))

            return a, c, ac, error_a_neg, error_a_pos, error_c_neg, error_c_pos, error_ac_neg, error_ac_pos, rho_new

        if self.disloctype == 'loop':

            ac, new_rho = self.calcRho(a1, a2, rho, printResult=printResult)

            error = []

            for item in combilist:
                error_ac, error_rho = self.calcRho(item[0], item[1], rho, printResult=False)
                error.append(error_ac)

            errorneg = ac - np.min(error)
            errorpos = np.abs(np.max(error) - ac)

            if printResult == True:
                print('a proportion: {0:.3f}'.format(ac))
                print('-: {0:.3f}'.format(errorneg))
                print('+: {0:.3f}'.format(errorpos))
                print('newrho: {0:.3f}'.format(rho))

            return ac, errorneg, errorpos, new_rho

        if self.disloctype == 'both':

            a, c, ac, ac, a_loop, rho_new = self.calcRho(a1, a2, rho, printResult=printResult)

            error_a_list = [];
            error_c_list = [];
            error_ac_list = [];
            error = [];

            for item in combilist:
                error_a, error_c, error_ca, error_aloop, error_rho = self.calcRho(item[0], item[1], rho,
                                                                                  printResult=False)
                error_a_list.append(error_a)
                error_c_list.append(error_c)
                error_ac_list.append(error_ca)
                error.append(error_aloop)

            error_a_neg = a - np.min(error_a_list);
            error_a_pos = np.abs(np.max(error_a_list) - a);
            error_c_neg = c - np.min(error_c_list);
            error_c_pos = np.abs(np.max(error_c_list) - c);
            error_ac_neg = ac - np.min(error_ac_list);
            error_ac_pos = np.abs(np.max(error_ac_list) - ac);
            errorneg = a_loop - np.min(error);
            errorpos = np.abs(np.max(error) - a_loop)

            if printResult == True:
                print('<a> loop proportion: {0:.3f}'.format(a_loop))
                print('-: {0:.3f}'.format(errorneg))
                print('+: {0:.3f}'.format(errorpos))
                print('<a> proportion: {0:.4f}'.format(a))
                print('-: {0:.3f}'.format(error_a_neg))
                print('+: {0:.3f}'.format(error_a_pos))
                print('<c> proportion: {0:.4f}'.format(c))
                print('-: {0:.3f}'.format(error_c_neg))
                print('+: {0:.3f}'.format(error_c_pos))
                print('<c+a> proportion: {0:.4f}'.format(ac))
                print('-: {0:.3f}'.format(error_ac_neg))
                print('+: {0:.3f}'.format(error_ac_pos))
                print('New rho: {0:.3f} 10^14 m^-2'.format(rho_new))

            return a, c, ac, error_a_neg, error_a_pos, error_c_neg, error_c_pos, error_ac_neg, error_ac_pos, a_loop, errorneg, errorpos, new_rho

    def calcEllipticity(self):
        ''' Calculate the a1 and a2 values for a given ellipticity, based on 30% of a loops with <11-20> and 
        70% of a loops with <10-10> habit planes, from literature values
        
        References
        ----------
        L. Balogh et al., Contrast factors of irradiation-induced dislocation loops in hexagonal materials
        
        '''

        data = np.loadtxt('data/ellipticities.txt')

        ellipticityList = data[:, 0]
        Chk0alList = data[:, 1]
        a1aLList = data[:, 2]
        a2aLList = data[:, 3]

        if np.min(ellipticityList) <= self.ellipticity <= np.max(ellipticityList):
            index = np.argmin(np.abs(self.ellipticity - ellipticityList))
            self.Chk0al = Chk0alList[index]
            self.a1aL = a1aLList[index]
            self.a2aL = a2aLList[index]
        else:
            raise ValueError('Ellipticity value must be in range 0.05 -> 20')


def extractDataDir(directory, calcLattice=True, calcLoop=True, ellipticity=1, wavelength=None):
    '''Extract physical paramaters from all CMWP solution files in a directory
    
    Paramaters
    -----------
    directory: str
        Directory containing solution files to parse.
    calcLattice: bool
        If true, calculate lattice params for phase 0.
    calcLoop: bool
        If true, calculate <a> and <c> loop fractions from a1, a2.
    ellipticity: float
        The ellipticity to use for <a>/<c> loop fraction calculation.
    wavelength: float
        If specified, use this wavelength instead of one determined from .sol file.
    
    Example
    -------
    >> data_df = extractDataDir('/home/rhys/Documents/CMWP-211102/2020_11_DESY/hyd/*.sol', calcLoop=False, ellipticity=1)
    '''

    file_list = []
    for file in glob.glob(directory):
        if ("int") not in file:
            file_list.append(file)

    file_list.sort()
    print('Parsing {0} solution files...'.format(len(file_list)))

    num_error = 0

    data_df = pd.DataFrame()

    for i, fullname in enumerate(file_list):

        with open(fullname, 'r', encoding='ISO-8859-1') as file_:

            dataDict = {}

            dataDict['filename'] = fullname.split('/')[-1]
            
            dataDict['peak_names'] = [];
            dataDict['peak_phase'] = [];

            lines = file_.readlines()

            # If the file ends with '*** END' then it means there was an error
            if lines[-1].startswith('*** END') == False:
                num_error += 1

            else:
                print('Reading {0}                     \r'.format(fullname), end='')

                readParams = 0
                ####### READ PHYSICAL PARAMATERS
                for line in lines:
                    if 'WSSR=' in line[0:5]: dataDict['wssr'] = float(line.split('=')[1])

                    ### BETWEEN HERE
                    if 'The final parameters:' in line:
                        readParams = 1

                    if (readParams == 1) and ('=' in line):
                        if line.split('=')[1] == '\n':
                            ignore=0
                        else:
                            dataDict[line.split('=')[0]] = float(line.split('=')[1][:-1])

                    ### AND HERE
                    if 'And now listing the physical' in line:
                        readParams = 0
                
                ####### READ PHYSICAL PARAMATER ERRORS
                for line in lines:
                    if 'error estimates for parameter' in line:
                        par = line.split(' ')[6]
                        textlist = line.split(':')[2].split(',')
                        error_abs = [None, None]
                        for idx, text in enumerate(textlist):
                            if 'e' in text:
                                nums = re.findall(r"[-+]?\d*\.?\d+|\d+", text)
                                error_abs[idx] = float(nums[0]) * 10 ** (int(nums[1]))
                            else:
                                error_abs[idx] = re.findall(r"[-+]?\d*\.?\d+|\d+", text)[0]

                        dataDict[par + '_neg'] = np.abs((float(error_abs[0]) / 100) * dataDict[par])
                        dataDict[par + '_pos'] = np.abs((float(error_abs[1]) / 100) * dataDict[par])

                ####### READ PEAK PHASE AND PEAK NAMES
                    if 'Found peak at' in line:
                        # Found peak at 6.5540, intensity=0, phase=1.
                        dataDict['peak_phase'].append(int(line.split('=')[-1][0]))

                    if 'hkl=' in line:
                        dataDict['peak_names'].append(line.split('=')[1][0:3])
                            
                 ####### READ PEAK POSITION AND INTENSITY
                dataDict['peak_pos'] = np.zeros(len(dataDict['peak_names']))
                dataDict['peak_int'] = np.zeros(len(dataDict['peak_names']))

                for line in lines:
                    if 'i_s0' in line:
                        if '+' in line:
                            dataDict['peak_pos'][int(line[5:9])] = float(line.split('=')[1].split('+')[0])

                    # EXAMPLE: i_max_7         = 939.718          +/- 4.919        (0.5234%)
                    # EXAMPLE: i_max_1         = 1.00679e+12      +/- 2.881e+09    (0.2861%)
                    if 'i_max' in line:
                        # 1.00679e+12
                        numberString = line.split('=')[-1].split('+/-')[0]
                        dataDict['peak_int'][int(line[6:10])] = float(numberString)

                    if (wavelength == None) and ('The wavelength is:' in line):
                        wavelength = float(line.split(':')[1].split('n')[0])

            ####### CALCULATE LATTICE PARAMS FOR PHASE 1
                if calcLattice == True:
                    
                    names = np.array(dataDict['peak_names'])
                    phases = np.array(dataDict['peak_phase'])
                    poss = np.array(dataDict['peak_pos'])
                    
                    res = calculateLatticeParams(names[(phases == 0) & (poss > 0)], 
                                                 poss[(phases == 0) & (poss > 0)], 
                                                 wavelength,
                                                 plotResult=False)
                                                      
                    dataDict['a_lat'] = res.params['a'].value * 10;
                    dataDict['a_lat_err'] = res.params['a'].stderr * 10;
                    dataDict['c_lat'] = res.params['c'].value * 10;
                    dataDict['c_lat_err'] = res.params['c'].stderr * 10

                    dataDict['covera'] = dataDict['c_lat'] / dataDict['a_lat']
                    dataDict['covera_err'] = dataDict['covera'] * np.sqrt(
                        (dataDict['c_lat_err'] / dataDict['c_lat']) ** 2 + (dataDict['a_lat_err'] / dataDict['a_lat']) ** 2)
                    dataDict['cellvol'] = dataDict['a_lat'] ** 2 * np.sin(np.deg2rad(60)) * dataDict['c_lat']
                    dataDict['cellvol_err'] = dataDict['cellvol'] * np.sqrt(
                        (dataDict['c_lat_err'] / dataDict['c_lat']) ** 2 + 2 * (dataDict['a_lat_err'] / dataDict['a_lat']) ** 2)

            ####### CALCULATE <a>/<c> LOOP FRACTIONS
                if calcLoop == True:
                    dataDict['a_loop'], dataDict['a_loop_neg'], dataDict['a_loop_pos'], dataDict[
                        'd_new'] = dislocationTypeCalc(
                        disloctype='loop', ellipticity=ellipticity).calcRhoError(
                        (dataDict['a1'], dataDict['a1_neg'], dataDict['a1_pos']),
                        (dataDict['a2'], dataDict['a2_neg'], dataDict['a2_pos']), dataDict['d'], printResult=False)
                    dataDict['d_new_neg'] = (dataDict['d_neg'] / dataDict['d']) * dataDict['d_new']
                    dataDict['d_new_pos'] = (dataDict['d_pos'] / dataDict['d']) * dataDict['d_new']

                data_df = data_df.append(dataDict, ignore_index=True)

    print(
        'Parsed {0} solution files succesfully                                                                '.format(
            data_df.shape[0]))
    if num_error > 0:
        print(
            '"*** END" not present in {0}                                                                     '.format(
                num_error))

    print('Done!')

    return data_df


def getBaseline(xvals, yvals, baseline, baseline_interpolate, baseline_interpolate_factor, write_to=None):
    """
    Function to create a background spline for CMWP.

    Parameters
    ----------
    xvals: np.array(float)
        Data x axis, i.e. 2theta.
    yvals: np.array(float)
        Data y axis, i.e. intensities.
    baseline: list(float)
        X positions from which to calculate background spline from data.
    baseline_interpolate: list(float)
        Additional positions for spline (not calculated from data!)
    baseline_interpolate_factor: list(float)
        Multiplication factor for baseline_interpolate values.
    write_to: str
        Location to save .bg-spline.dat file.

    Returns
    -------
    baseline_pos: np.array(float)
        Positions for background spline.
    baseline_int: np.array(float)
        Intensities for background spline.
    cs: CubicSpline
        Cubic spline for background.

    """
    baseline_pos = baseline[:]

    # Get index of baseline position
    num_index = (np.abs(xvals[:, None] - baseline_pos)).argmin(axis=0)

    # Refine peak position by searching in +-searchrange
    searchmask = np.linspace(num_index - 5, num_index + 5, 2 * 5 + 1, axis=1).astype(int)
    baseline_int = np.mean(yvals[:, None][searchmask], axis=1).T[0]-0.8

    cs = CubicSpline(baseline_pos, baseline_int)

    # Add additional baseline_interpolate points
    if baseline_interpolate is not None:
        baseline_pos += baseline_interpolate
        baseline_int = np.append(baseline_int, cs(baseline_interpolate)*baseline_interpolate_factor)

    # Sort values
    idx = np.argsort(baseline_pos)
    baseline_pos = np.array(baseline_pos)[idx]
    baseline_int = np.array(baseline_int)[idx]

    if write_to is not None:
        with open(write_to, 'w+') as f:
            np.savetxt(fname=f, X=np.transpose([baseline_pos, baseline_int]), fmt=('%1.3f'))

    cs = CubicSpline(baseline_pos, baseline_int)

    return baseline_pos, baseline_int, cs


def getPeaks(xvals, yvals, peak_pos, peak_name, cs, searchrange, phasenumber = 0, ax=None, plotcolour=None, mode='w', write_to=None):
    """

    Parameters
    ----------
    xvals: np.array(float)
        Data x axis, i.e. 2theta.
    yvals: np.array(float)
        Data y axis, i.e. intensities.
    peak_pos: list(float)
        Theoretical peak positions.
    peak_name: list(str)
        Peak hkls.
    cs: CubicSpline
        Background cubic spline (for calculating actual peak intensity).
    searchrange: int
        Range to search for maxima.
    phaseNumber: int
        Phase number to use for peak-index.
    ax: Ax
        Axes to plot lines and hkl.
    plotcolour: str
        Colour to plot lines.
    mode: str
        Write mode to file i.e. 'w' or 'w+'.
    write_to: str
        Location to save .peak-index.dat file.

    Returns
    -------
    peak_pos, peak_name, peak_int
        Caulcated peak positions, hkls and intensities

    """
    # Remove any peaks outside the range of the data
    mask = np.logical_and(peak_pos < np.max(xvals), peak_pos > np.min(xvals))
    peak_name = peak_name[mask]
    peak_pos = peak_pos[mask]

    # Get approx index of peak position
    approx_peak_index = (np.abs(xvals[:, None] - peak_pos)).argmin(axis=0)

    # Refine peak position by searching in +-searchrange
    searchmask = np.linspace(approx_peak_index - searchrange, approx_peak_index + searchrange, 2 * searchrange + 1,
                             axis=1).astype(int)
    peak_index = np.argmax(yvals[:, None][searchmask], axis=1).T[0] + approx_peak_index - searchrange

    # Update peak_pos and peak_int
    peak_pos = xvals[peak_index]
    peak_int = yvals[peak_index] 

    if write_to is not None:
        with open(write_to, mode) as f:
            np.savetxt(fname=f, X=np.c_[peak_pos, peak_int - cs(peak_pos), peak_name], fmt='%s %s %s ' + str(phasenumber))
            
    if ax is not None:
        ax.vlines(peak_pos, ymin=np.min(yvals), ymax=np.max(yvals), alpha=0.1, colors=plotcolour)
    
        for pos, intensity, name in zip(peak_pos, peak_int+10, peak_name):
            ax.text(pos, intensity, name, horizontalalignment = 'center', c=plotcolour)


    return peak_pos, peak_name, peak_int - cs(peak_pos)
