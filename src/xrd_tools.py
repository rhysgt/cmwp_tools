import numpy as np

def getReflections(crystalType, wavelength, outputType='2theta', a=None, b=None, c=None, alpha=90., beta=90., gamma=90.,
                   printReflections=False):
    """ Makes a list of hkl and peak position (2theta, d or k) for a given wavelength and lattice.
    
    Params
    ------
    crystalType: str {hcp, bcc, fcc, fct or cubic}
        Crystal structure.
    wavelength: float
        Wavelength in angstrom.
    outputType: str {2theta, d, k}
        X axis type.
    a, b, c, alpha, beta, gamma: float
        Lattice paramaters in angstrom/ degrees.
    printReflections: bool
        If true, print a list of peak names and positions.
        
    Returns
    -------
    peak_name: numpy.ndarray(str)
        Peak hkls.
    peak_pos: numpy.ndarray(float)
        Positions of peaks in 2theta, d [Å] or k [Å]
    """
    
    # Peak indexes for different crystal structures
    hkls = {'hcp': ['100', '002', '101', '102', '110', '013', '020', '112', '021', '004',
                   '022', '014', '023', '210', '211', '114', '212', '015', '024', '030',
                   '213', '032', '006', '025', '016', '214', '220', '310', '222', '311',
                   '116', '034', '312', '215', '026', '017', '313', '040', '041', '224'],
            
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
                     '424', '610', '532', '026', '344', '145', '335', '262', '452', '316'],
            
           'fct': ['110', '111', '002', '200', '112', '022', '220', '113', '311', '222', '004', '400'],
           
           'orth': ['011', '111', '002', '020', '200', '211', '022', '202', '220', '013',
                   '113', '031', '131', '311', '222', '213', '004', '231', '040', '400',
                   '033', '133', '024', '313', '204', '331', '042', '402', '240', '420',
                   '224', '242', '422', '115', '333', '151']
           }
    
    peak_pos_2th = []; peak_pos_d = []; peak_pos_k = []; peak_name = []
    
    # Check for errors
    if outputType not in ['2theta', 'd', 'k']:
        raise Exception('outputType must be 2theta, d or k')

    if crystalType in ['hcp', 'fct']:
        if (a == None) or (c == None):
            raise Exception('a and c lattice paramaters must be specified')
        else:
            b = a
    elif crystalType in ['bcc', 'fcc', 'cubic']:
        if a == None:
            raise Exception('a lattice paramater must be specified')
        else:
            b = c = a
    elif crystalType in ["orth"]:
        if (a==None) or (b==None) or (c==None):
            raise Exception('all a, b & c lattice parameters must be specified')
    else:
        raise Exception('crystalType of hcp, bcc, fcc, fct or cubic must be specified')
    
    # Calculate reflections
    for peak in hkls[crystalType]:
        h = int(peak[0]); k = int(peak[1]); l = int(peak[2]);
        if crystalType == 'hcp':
            d = (np.sqrt(1/((4/3)*(((h*h)+(k*k)+h*k)/(a*a))+(l*l/(c*c)))))
        elif crystalType == 'bcc' or 'fcc' or 'cubic' or 'orth':
            X = np.sqrt(1 - np.cos(np.radians(alpha))**2 - np.cos(np.radians(beta))**2 - np.cos(np.radians(gamma))**2 +
                                    2*np.cos(np.radians(alpha))*np.cos(np.radians(beta))*np.cos(np.radians(gamma)))
            Y = np.sqrt(((h/a)**2)*np.sin(np.radians(alpha))**2 + 
                                    ((k/b)**2)*np.sin(np.radians(beta))**2 +
                                    ((l/c)**2)*np.sin(np.radians(gamma))**2 - 
                                    ((2*k*l)/(b*c))*(np.cos(np.radians(alpha))-np.cos(np.radians(beta))*np.cos(np.radians(gamma))) - 
                                    ((2*l*h)/(c*a))*(np.cos(np.radians(beta))-np.cos(np.radians(gamma))*np.cos(np.radians(alpha))) - 
                                    ((2*h*k)/(a*b))*(np.cos(np.radians(gamma))-np.cos(np.radians(alpha))*np.cos(np.radians(beta))))
            d = X/Y     
        if wavelength/(2*d) < 1:
            in_rad = 2*np.arcsin(wavelength/(2*d))
            peak_pos_2th.append(np.rad2deg(in_rad))
            peak_pos_d.append(d)
            peak_pos_k.append(1/d)
            peak_name.append(peak)

    # Print reflections
    if printReflections:
        if crystalType in ['hcp', 'fct']:
            print(r'Indexing {0} peaks for a = {1:.3f} Å, c = {2:.3f} Å at wavelength = {3:.3f} Å:'.format(crystalType,a,c,wavelength))
                
        elif crystalType in ['bcc', 'fcc', 'cubic']:
            print(r'Indexing {0} peaks for a = {1:.3f} Å: at wavelength = {2:.3f} Å'.format(crystalType, a, wavelength))
            
        print('')
        print('index\td (Å)\tK (1/Å)\t2theta')
        for name, pos_2th, pos_d, pos_k in zip(peak_name, peak_pos_2th, peak_pos_d, peak_pos_k):
            print('{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(name, pos_d, pos_k, pos_2th))
        print('\n')

    # Output reflections
    if outputType == '2theta':
        return np.array(peak_name), np.array(peak_pos_2th)
    elif outputType == 'd':
        return np.array(peak_name), np.array(peak_pos_d)
    elif outputType == 'k':
        return np.array(peak_name), np.array(peak_pos_k)