'''                                                                                                                                     
get a color from a range                                                                                                                
'''

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import matplotlib.cm as cmx
from matplotlib import rc



class colourFromRange():

    def __init__(self, indexRange, cmap='Reds'):
    
        

        #For aesthetics      
        self.cmap=cmap
        cm = plt.get_cmap(cmap)
        cNorm  = colors.Normalize(vmin=indexRange[0], \
                            vmax=indexRange[1])
        self.scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
	#####                                                                                                                           

    def __getitem__(self, index):
        return self.scalarMap.to_rgba(index)

