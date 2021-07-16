import os, re, ast, glob
import numpy as np
import fabio
from scipy import optimize
from lmfit import minimize, Parameters, fit_report
import matplotlib.pyplot as plt
import pandas as pd

def getReflections(crystalType, wavelength, outputType='2theta', a=None, c=None, printReflections=False):
    """ Makes a list of hkl and 2theta for a given wavelength and lattice.
    
    Params
    ------
    crystalType: str {hcp, bcc, fcc or cubic}
        Crystal structure.
    wavelength: float
        Wavelength in angstrom.
    outputType: str {2theta, d, k}
        X axis type.
    a, c: float
        Lattice paramaters in angstrom.
    printReflections: bool
        If true, print a list of peak names and positions.
        
    Returns
    -------
    peak_name: numpy.ndarray(str)
        Peak hkls.
    peak_pos: numpy.ndarray(float)
        2theta positions of peaks.
    """
    
    hkls = {'hcp': ['100','002','101','102','110','013','020','112','021','004',
                   '022','014','023','210','211','114','212','015','024','030',
                   '213','032','006','025','016','214','220','310','222','311',
                   '116','034','312','215','026','017','313','040','041','224'],
            
           'bcc': ['101', '200', '211', '022', '103', '222', '213', '040', '330', '420', 
                   '332', '422', '134', '521', '404', '433', '244', '352', '026', '154', 
                   '226', '163', '444', '701', '406', '633', '426', '073', '651', '080', 
                   '174', '464', '563', '606', '374', '626', '752', '048', '383', '248'],
            
           'fcc': ['111', '020', '220', '311', '222', '400', '313', '420', '224', '333', 
                   '044', '531', '060', '062', '353', '226', '444', '171', '064', '624', 
                   '371', '080', '373', '446', '066', '157', '266', '480', '375', '284', 
                   '466', '913', '844', '177', '086', '268', '377', '666', '359', '846'],

           'cubic': ['100', '101', '111', '200', '210', '112', '220', '122', '310', '131', 
                     '222', '032', '132', '400', '232', '411', '313', '024', '241', '332', 
                     '224', '403', '431', '511', '324', '125', '440', '414', '343', '513', 
                     '424', '610', '532', '026', '344', '145', '335', '262', '452', '316']}
    
    peak_pos_2th = []; peak_pos_d = []; peak_name = []
    
    # Check for errors
    if crystalType == 'hcp':
        if (a == None) or (c == None):
            raise Exception('a and c lattice paramaters must be specified')
    elif crystalType == 'bcc' or 'fcc' or 'cubic':
        if a == None:
            raise Exception('a lattice paramater must be specified')
    else:
        raise Exception('crystalType of hcp, bcc, fcc or cubic must be specified')
    
    # Calculate reflections
    for peak in hkls[crystalType]:
        h = int(peak[0]); k = int(peak[1]); l = int(peak[2]);
        if crystalType == 'hcp':
            d = (np.sqrt(1/((4/3)*(((h*h)+(k*k)+h*k)/(a*a))+(l*l/(c*c)))))
        elif crystalType == 'bcc' or 'fcc' or 'cubic':
            d = a / np.sqrt(h*h+k*k+l*l)
        if wavelength/(2*d) < 1:
            in_rad = 2*np.arcsin(wavelength/(2*d))
            peak_pos_2th.append(np.rad2deg(in_rad))
            peak_pos_d.append(d)
            peak_name.append(peak)

    # Print reflections
    if printReflections:
        if crystalType == 'hcp':
            print('Indexing hcp peaks for a = {0:.3f} A, c = {1:.3f} A at wavelength = {2:.3f} A:'.format(a,c, wavelength))
            
        elif crystalType == 'bcc' or 'fcc' or 'cubic':
            print('Indexing {0} peaks for a = {1:.3f} A: at wavelength = {2:.3f} A'.format(crystalType, a, wavelength))
        for name, pos_2th, pos_d in zip(peak_name, peak_pos_2th, peak_pos_d):
            print('{0}\t{1:.3f}\t{2:.3f}'.format(name,pos_d, pos_2th))

    if outputType == '2theta':
        return np.array(peak_name), np.array(peak_pos_2th)
    elif outputType == 'd':
        return np.array(peak_name), np.array(peak_pos_d)
    elif outputType == 'k':
        return np.array(peak_name), 1/np.array(peak_pos_d)

def load_tifs(filename): 
    """ Load tiff file using fabio.
    
    Params
    -------
    filename: str
        Path to tiff file.
    
    """
    if isinstance(filename, list): #a list of filenames is given: return the average (mean) image as a numpy array
        data=[]
        for f in filename:
            try:
                data.append(fabio.open(f).data)
            except:
                print('Warning: file %s skipped (error while opening image with fabio)'%f)
        assert len(data)>0, 'Error in pf_int.load_tif(): no data found!'
        data = np.mean(np.array(data), axis = 0)
    else: #just one image
        assert os.path.isfile(filename), 'Error in pf_int.load_tif(): filename does not exist!'
        data=fabio.open(filename).data    
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
        int1 = ai1.integrate1d(file1, 10000,correctSolidAngle=True, polarization_factor=0.99, method='full_csr', unit='2th_deg')
        int2 = ai2.integrate1d(file2, 10000,correctSolidAngle=True, polarization_factor=0.99, method='full_csr', unit='2th_deg')
    else:
        int1 = ai1.integrate1d(file1, 10000,correctSolidAngle=True, azimuth_range=azimuth, polarization_factor=0.99, method='csr', unit='2th_deg')
        int2 = ai2.integrate1d(file2, 10000,correctSolidAngle=True, azimuth_range=azimuth, polarization_factor=0.99, method='csr', unit='2th_deg')
    
    # Find limit index
    limits[0] = int(np.abs(int1[0] - limits[0]).argmin())
    limits[1] = int(np.abs(int2[0] - limits[1]).argmin())
    
    # Find cutoff index
    cutoff_index = [0,0]
    if isinstance(cutoffs, float):
        cutoff_index[0] = int(np.abs(int1[0] - cutoffs).argmin())
        cutoff_index[1] = int(np.abs(int2[0] - cutoffs).argmin())+1
    else:
        cutoff_index[0] = int(np.abs(int1[0] - cutoffs[0]).argmin())
        cutoff_index[1] = int(np.abs(int2[0] - cutoffs[1]).argmin())+1
    

    # Find shift between detectors where they join
    shift = np.mean(int2[1][cutoff_index[1]:cutoff_index[1]+50] - int1[1][cutoff_index[0]-50:cutoff_index[0]])
    
    intnew = np.array([np.concatenate((int1[0][limits[0]:cutoff_index[0]], int2[0][cutoff_index[1]:limits[1]])),
                       np.concatenate((int1[1][limits[0]:cutoff_index[0]], int2[1][cutoff_index[1]:limits[1]]-shift))])

    # Add offset to make numbers positive
    intnew[1] -= np.min(intnew[1])
    intnew[1]+=10

    return intnew

def calculateObjective(params, hkls, poss, wavelength):
    """
    This is the objective function based on the Hull-Davey equation 
    to calculate lattice paramaters.
    
    """
    vals = params.valuesdict()
    a = vals['a']
    c = vals['c']

    objective=[]
    for hkl, pos in zip(hkls, poss):
        h=int(hkl[0]); k=int(hkl[1]); l=int(hkl[2]);
        K = 2 * np.sin(np.deg2rad(pos/2)) / wavelength
        logd2 = 2 * np.log(1/K)
        eq = 2*np.log(a)-np.log((4/3)*(h**2+h*k+k**2)+l**2/(c/a)**2)
        weight = np.tan(np.deg2rad(pos/2))
        diff = (logd2 - eq)**2 * weight**2

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
    fit_params.add('a', value = 0.3232, min=0.31, max=0.34)
    fit_params.add('c', value = 0.5147, min=0.50, max=0.53)
    
    res = minimize(calculateObjective, params = fit_params, args=(hkls, poss), kws={'wavelength':wavelength})
    
    if printResult:
        print(fit_report(res))
        
    
        
    if plotResult == True:
        a=res.params['a'].value
        c=res.params['c'].value
        fig, ax = plt.subplots(figsize=(10,5))
        for hkl, pos in zip(hkls, poss):
            h=int(hkl[0]); k=int(hkl[1]); l=int(hkl[2]);
            K = 2 * np.sin(np.deg2rad(pos/2)) / wavelength
            logd2 = 2 * np.log(1/K)
            eq = 2*np.log(a)-np.log((4/3)*(h**2+h*k+k**2)+l**2/(c/a)**2)
            weight = np.tan(np.deg2rad(pos/2))
            diff = (logd2 - eq)**2 * weight**2
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
            
        self.a_param = 0.3232; self.c_param = 0.5147;
        self.b = 0.3232; self.Chk0 = 0.25;
        self.hkls = ['100', '002', '101', '102', '110', '200', '112', '201', '004', '202', 
               '104', '203', '210', '211', '114', '105', '204', '300', '213', '302', '006']
        
        if self.disloctype == 'def' or self.disloctype == 'both':

            self.Chk0a = 0.2368; self.a1a = -0.0436; self.a2a = -0.3011; self.ba = 0.3232;
            self.Chk0c = 0.045; self.a1c = 5.8513; self.a2c = 3.1882; self.bc = 0.5140;
            self.Chk0ca = 0.0897; self.a1ca = 2.4520; self.a2ca = 0.7474; self.bca = 0.6078;
            
        if self.disloctype == 'loop' or self.disloctype == 'both':
            
            self.Chk0cL = 0.0815; self.a1cL = 2.6411; self.a2cL = 1.0102;
            self.baL = 0.3232; self.bcL = 0.2790;
            
            if ellipticity == None:
                raise ValueError('Please specify ellipticity')
            else:
                self.ellipticity=ellipticity
                self.calcEllipticity()

    def calcH1H2(self, hkl):

        h = int(hkl[0]); k = int(hkl[1]); l = int(hkl[2])
        h1_squared = (h**2 + k**2 + (h + k)**2)*l**2 / (h**2 + k**2 + (h+k)**2 + 3/2*(self.a_param/self.c_param)**2*l**2)**2
        h2_squared = l**4 / (h**2 + k**2 + (h + k)**2 + 3/2*(self.a_param/self.c_param)**2*l**2)**2
        return h1_squared, h2_squared

    def calcEpsSquaredExp(self, hkl, a1, a2, rho):

        h = int(hkl[0]); k = int(hkl[1]); l = int(hkl[2])
        h1_squared, h2_squared = self.calcH1H2(hkl)
        C = self.Chk0*(1+(a1*h1_squared)+(a2*h2_squared))
        EpsCalcSquared = rho*C*self.b**2
        return EpsCalcSquared

    def calcEpsSquaredThe(self, hkl, a_am=None, c_am=None, ca_am=None, RoaL=None, RocL=None):
        
        h = int(hkl[0]); k = int(hkl[1]); l = int(hkl[2])
        h1_squared, h2_squared = self.calcH1H2(hkl)
        
        EpsSquaredThe = 0
        
        if self.disloctype=='def' or self.disloctype == 'both':
            
            Ca = self.Chk0a*(1+self.a1a*h1_squared+self.a2a*h2_squared)
            Cc = self.Chk0c*(1+self.a1c*h1_squared+self.a2c*h2_squared)
            Cca = self.Chk0ca*(1+self.a1ca*h1_squared+self.a2ca*h2_squared)
            
            EpsSquaredThe += a_am*Ca*self.ba**2 + c_am*Cc*self.bc**2 + ca_am*Cca*self.bca**2

        if self.disloctype=='loop' or self.disloctype == 'both':

            CaLoop = self.Chk0al*(1+self.a1aL*h1_squared+self.a2aL*h2_squared)
            CcLoop = self.Chk0cL*(1+self.a1cL*h1_squared+self.a2cL*h2_squared)

            EpsSquaredThe += RoaL*CaLoop*self.baL**2 + RocL*CcLoop*self.bcL**2

        return EpsSquaredThe

    def minimisationfunct(self, params, a1, a2, rho):
        
        if self.disloctype == 'def':
            a_am, c_am, ca_am = params
            totlist = []  
            for hkl in self.hkls:
                DiffSquared = (self.calcEpsSquaredExp(hkl, a1, a2, rho) - self.calcEpsSquaredThe(hkl, a_am=a_am, c_am=c_am, ca_am=ca_am))
                totlist.append(DiffSquared)
    
        if self.disloctype == 'loop':
            RoaL, RocL = params
            totlist = []  
            for hkl in self.hkls:
                DiffSquared = (self.calcEpsSquaredExp(hkl, a1, a2, rho) - self.calcEpsSquaredThe(hkl, RoaL=RoaL, RocL=RocL))
                totlist.append(DiffSquared)
                
        if self.disloctype == 'both':
            a_am, c_am, ca_am, RoaL, RocL = params
            totlist = []  
            for hkl in self.hkls:
                DiffSquared = (self.calcEpsSquaredExp(hkl, a1, a2, rho) - self.calcEpsSquaredThe(hkl, a_am=a_am, c_am=c_am, ca_am=ca_am, RoaL=RoaL, RocL=RocL))
                totlist.append(DiffSquared)    

        return np.linalg.norm(totlist)

    def calcRho(self, a1, a2, rho, printResult):

        if printResult is True:
            print('a1 = {0:.4f}\na2 = {1:.4f}\nrho = {2:.4f}\n'.format(a1, a2, rho))
        
        if self.disloctype == 'def':
            res = optimize.minimize(self.minimisationfunct, x0=[rho/3, rho/3, rho/3], 
                                          args=(a1, a2, rho), bounds=((0.001, 200), (0.001, 200), (0.001, 200)), 
                                        method='SLSQP', options={'ftol': 1e-9, 'eps': 1e-9})

            a = res.x[0]/np.sum(res.x); c = res.x[1]/np.sum(res.x); ac = res.x[2]/np.sum(res.x)
            rho_new = np.sum(res.x)

            if printResult is True:
                print('<a> proportion: {0:.4f}'.format(a))
                print('<c> proportion: {0:.4f}'.format(c))
                print('<c+a> proportion: {0:.4f}'.format(ac))
                print('New rho: {0:.3f} 10^14 m^-2'.format(rho_new))

            return a, c, ac, rho_new
        
        if self.disloctype == 'loop':
            res = optimize.minimize(self.minimisationfunct, x0=[rho/2, rho/2], 
                                  args=(a1, a2, rho), bounds=((0.001, 200), (0.001, 200)), 
                                method='SLSQP', options={'ftol': 1e-9, 'eps': 1e-9})

            ac = res.x[0]/np.sum(res.x)
            rho_new = np.sum(res.x)

            if printResult is True:
                print('ellipticity = {0}\n'.format(self.ellipticity))
                print('<a> loop proportion: {0:.4f}'.format(ac))
                print('New rho: {0:.4f} 10^14 m^-2'.format(rho_new))

            return ac, rho_new
        
        if self.disloctype == 'both':
            res = optimize.minimize(self.minimisationfunct, x0=[rho/5, rho/5, rho/5, rho/5, rho/5], 
                                  args=(a1, a2, rho), bounds=((0.001, 200), (0.001, 200), (0.001, 200), (0.001, 200), (0.001, 200)), 
                                method='SLSQP', options={'ftol': 1e-9, 'eps': 1e-9})

            a = res.x[0]/np.sum(res.x); c = res.x[1]/np.sum(res.x); ac = res.x[2]/np.sum(res.x); a_loop = res.x[3]/np.sum(res.x); c_loop = res.x[4]/np.sum(res.x);
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
        
        combilist = [[a1-a1_neg, a2],[a1+a1_pos, a2],[a1, a2+a2_pos],[a1, a2-a2_neg],
                         [a1-a1_neg/2, a2-a2_neg/2],[a1+a1_pos/2, a2+a2_pos/2],[a1-a1_neg/2, a2+a2_pos/2],[a1+a1_pos/2, a2-a2_neg/2]]
        
        if self.disloctype == 'def':
            
            a, c, ac, rho_new = self.calcRho(a1, a2, rho, printResult=printResult)

            error_a_list=[];error_c_list=[];error_ac_list=[];

            for item in combilist:
                error_a, error_c, error_ca, error_rho = self.calcRho(item[0], item[1], rho, printResult=False)
                error_a_list.append(error_a)
                error_c_list.append(error_c)
                error_ac_list.append(error_ca)

            error_a_neg = a - np.min(error_a_list); error_a_pos = np.abs(np.max(error_a_list)-a);
            error_c_neg = c - np.min(error_c_list); error_c_pos = np.abs(np.max(error_c_list)-c);
            error_ac_neg = ac - np.min(error_ac_list); error_ac_pos = np.abs(np.max(error_ac_list)-ac);

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

            error=[]

            for item in combilist:
                error_ac, error_rho = self.calcRho(item[0], item[1], rho, printResult=False)
                error.append(error_ac)

            errorneg = ac - np.min(error)
            errorpos = np.abs(np.max(error)-ac)

            if printResult == True:
                print('a proportion: {0:.3f}'.format(ac))
                print('-: {0:.3f}'.format(errorneg))
                print('+: {0:.3f}'.format(errorpos))
                print('newrho: {0:.3f}'.format(rho))

            return ac, errorneg, errorpos, new_rho

        if self.disloctype == 'both':
            
            a, c, ac, ac, a_loop, rho_new = self.calcRho(a1, a2, rho, printResult=printResult)

            error_a_list=[];error_c_list=[];error_ac_list=[]; error=[];

            for item in combilist:
                error_a, error_c, error_ca, error_aloop, error_rho = self.calcRho(item[0], item[1], rho, printResult=False)
                error_a_list.append(error_a)
                error_c_list.append(error_c)
                error_ac_list.append(error_ca)
                error.append(error_aloop)

            error_a_neg = a - np.min(error_a_list); error_a_pos = np.abs(np.max(error_a_list)-a);
            error_c_neg = c - np.min(error_c_list); error_c_pos = np.abs(np.max(error_c_list)-c);
            error_ac_neg = ac - np.min(error_ac_list); error_ac_pos = np.abs(np.max(error_ac_list)-ac);
            errorneg = a_loop - np.min(error); errorpos = np.abs(np.max(error)-a_loop)

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
        
        ellipticityList = data[:,0]
        Chk0alList = data[:,1]
        a1aLList = data[:,2]
        a2aLList = data[:,3]

        if np.min(ellipticityList) <= self.ellipticity <= np.max(ellipticityList):
            index = np.argmin(np.abs(self.ellipticity-ellipticityList))
            self.Chk0al = Chk0alList[index]
            self.a1aL = a1aLList[index]
            self.a2aL = a2aLList[index]
        else: 
            raise ValueError('Ellipticity value must be in range 0.05 -> 20')

def extractDataDir(directory, ellipticity=1):
    '''Extract physical paramaters from all CMWP solution files in a directory
    
    '''    

    file_list = []
    for file in glob.glob(directory):
        if ("int") not in file:
            file_list.append(file)
    
    file_list.sort()
    print('Parsing {0} solution files...'.format(len(file_list)))
        
    data_df = extractDataFile(file_list, ellipticity=ellipticity)

    return data_df

def extractDataFile(files, ellipticity):
    '''Extract physical paramaters from CMWP solution files
    
    ''' 
   
    data_list=[]; index_list=[];
    num_error = 0
    
    for i, fullname in enumerate(files):
        
        with open(fullname, 'r', encoding='ISO-8859-1') as file_:
            
            lines = file_.readlines()
            
            # If the file ends with '*** END' then it means there was an error
            if lines[-1].startswith('*** END') == False:
                num_error += 1
                
            else:
                print('Reading {0}                     \r'.format(fullname), end='')
            
                for line in lines:
                    if 'WSSR=' in line[0:5]: wssr = float(line.split('=')[1])
                    if 'a1=' in line[0:3]: a1 = float(line.split('=')[1])
                    if 'a2=' in line[0:3]: a2 = float(line.split('=')[1])
                    if 'b=' in line[0:2]: b = float(line.split('=')[1]) 
                    if 'c=' in line[0:2]: c = float(line.split('=')[1])
                    if 'd=' in line[0:2]: d = float(line.split('=')[1])
                    if 'e=' in line[0:2]: e = float(line.split('=')[1])
                    if 'M=' in line[0:2]: m = float(line.split('=')[2])

                ## Catch errors
                for idx, param in enumerate([wssr, a1, a2, b, c, d, e, m]):
                    if param == None: raise ValueError('\033[91m' + 'Error with file {0}'.format(fullname) + '\033[0m')

                i=0
                peak_names = []; peak_pos = []; peak_int = [];
                for line in lines:
                    if 'error estimates for parameter' in line:
                        par=line.split(' ')[6]
                        textlist = line.split(':')[2].split(',')
                        error_abs = [None, None]
                        for idx, text in enumerate(textlist):

                            if 'e' in text:
                                nums = re.findall(r"[-+]?\d*\.?\d+|\d+", text)
                                error_abs[idx] = float(nums[0]) * 10**(int(nums[1]))
                            else:
                                error_abs[idx] = re.findall(r"[-+]?\d*\.?\d+|\d+", text)[0]

                        ## Calculate absolute error values
                        listpar=['a1', 'a2', 'b', 'c', 'd', 'e']

                        if par == listpar[0]:
                            a1_neg = np.abs((float(error_abs[0])/100)*a1); a1_pos = np.abs((float(error_abs[1])/100)*a1);
                        if par == listpar[1]:
                            a2_neg = np.abs((float(error_abs[0])/100)*a2); a2_pos = np.abs((float(error_abs[1])/100)*a2);
                        if par == listpar[2]:
                            b_neg = np.abs((float(error_abs[0])/100)*b); b_pos = np.abs((float(error_abs[1])/100)*b);
                        if par == listpar[3]:
                            c_neg = np.abs((float(error_abs[0])/100)*c); c_pos = np.abs((float(error_abs[1])/100)*c);
                        if par == listpar[4]:
                            d_neg = np.abs((float(error_abs[0])/100)*d); d_pos = np.abs((float(error_abs[1])/100)*d);
                        if par == listpar[5]:
                            e_neg = np.abs((float(error_abs[0])/100)*e); e_pos = np.abs((float(error_abs[1])/100)*e);
                            m_neg = np.abs((float(error_abs[0])/100)*m); m_pos = np.abs((float(error_abs[1])/100)*m);

                    if 'hkl=' in line:
                        peak_names.append(line.split('=')[1][0:3])
                    if 'i_s0' in line:
                        if '+' in line:
                            peak_pos.append(float(line.split('=')[1].split('+')[0]))
                            
                    # EXAMPLE: i_max_7         = 939.718          +/- 4.919        (0.5234%)
                    # EXAMPLE: i_max_1         = 1.00679e+12      +/- 2.881e+09    (0.2861%)
                    if 'i_max' in line:
                        # 1.00679e+12
                        numberString = line.split('=')[-1].split('+/-')[0]                     

                        if 'e' in numberString:
                            # [1.00679, 12]
                            num, power = numberString.split('e+')
                            peak_int.append(float(num) * 10**int(power))
                        else:
                            peak_int.append(float(numberString))
                            
                            
                    if 'The wavelength is:' in line:
                        wavelength_sel = float(line.split(':')[1].split('n')[0])
                    

                a_lat = np.nan; a_err = np.nan; c_lat = np.nan; c_err = np.nan;

                if (peak_names != []) and (peak_pos != []):
                    res = calculateLatticeParams(peak_names, peak_pos, wavelength_sel, plotResult=False)
                    a_lat=res.params['a'].value*10; a_err=res.params['a'].stderr*10; c_lat=res.params['c'].value*10; c_err=res.params['c'].stderr*10

                a_loop, a_loop_neg, a_loop_pos, d_new =  dislocationTypeCalc(
                    disloctype='loop', ellipticity=ellipticity).calcRhoError(
                    (a1, a1_neg, a1_pos), (a2, a2_neg, a2_pos), d, printResult=False)

                if (peak_names != []) and (peak_pos != []):

                    index_list.append(i)
                    data_list.append([fullname.split('/')[-1]] + [np.round(x ,6) for x in [wssr, d, d_new, (d_neg/d)*d_new, (d_pos/d)*d_new, a1, a1_neg, a1_pos, a2, a2_neg, a2_pos, 
                                      m, m_neg, m_pos, a_loop, a_loop_neg, a_loop_pos, b, b_neg, b_pos, c, c_neg, c_pos, a_lat, c_lat, a_err, c_err]])
    
    print('Parsed {0} solution files succesfully                                                                    '.format(len(index_list)))
    if num_error >0:
        print('"*** END" not present in {0}                                                                     '.format(num_error))
    
    data_df = pd.DataFrame(np.array(data_list), 
                           columns=['filename', 'wssr', 'rho', 'rho_new', 'rho_neg', 'rho_pos', 'a1', 'a1_neg', 'a1_pos', 'a2', 'a2_neg', 'a2_pos', 
                                    'm', 'm_neg', 'm_pos', 'a_loop', 'a_loop_neg', 'a_loop_pos', 'b', 'b_neg', 'b_pos', 'c', 'c_neg', 'c_pos', 'a_lat', 'c_lat', 'a_err', 'c_err'], 
                           index=index_list)
    
    data_df.reset_index(inplace=True)
    data_df.drop('index', axis=1, inplace=True)
    
    for col in data_df.columns[1:]:
        data_df[col] = pd.to_numeric(data_df[col], downcast="float")
    
    print('Done!')
    
    return data_df

