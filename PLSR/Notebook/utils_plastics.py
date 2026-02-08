
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotWavelength(sample,n,ax=None,cbar=True):
    '''
    Plot image at a given wavelength.

    Parameters
    ----------
    sample : numpy array
        multispectral image
    
    n : int
        nb of wavelength to show
        
     ax : matplotlib.axes._subplots.AxesSubplot, default=None
        add the plot to an existing axis
        
    cbar : bool
        wether to display colorbar or not

    Returns
    -------
    None

    '''
    if ax is None:
        ax=plt.gca()
    wave=sample[:,:,n]
    img=ax.imshow(wave,aspect='auto',cmap='Greys')
    if cbar:
        plt.colorbar(img)
    ax.set(title=f"{n}th wavelength")

def plotSpectra(wvgood, sample=None,row=None,col=None,ax=None,wavelength=None,**plt_kwargs):
    '''
    Plot pixel's spectra.

    Parameters
    ----------
    wvgood : numpy array
        values of the wavelengths
    
    sample : numpy array
        multispectral image
    
    row, col : int 
        row and column of the pixel
        
    wavelength : numpy array
        give directly the wavelength spectra array, instead of wvgood and sample
        
    **plt_kwargs : matplotlib plot arguments
     
    Returns
    -------
    None

    '''
    if ax is None:
        ax=plt.gca()
    if wavelength is not None:
        ax.plot(wvgood.flatten(),wavelength,**plt_kwargs)
    else:
        spectra=sample[row,col,:]
        ax.plot(wvgood.flatten(),spectra,**plt_kwargs)
        ax.set(xlabel="Wavelength (nm)",ylabel="Value",title=f"Spectra pixel ({row},{col})")
    
def imshow(f):
    '''
    Custom imshow.

    Parameters
    ----------
    f : array
     
    Returns
    -------
    None

    '''
    plt.figure()
    plt.imshow(f,cmap='gray',aspect='auto')
    
def plotPCA2d(ds,n_components=2,axes=[1,2],ax=None,labels=None,arrows=True,width_arrow=0.01,**plt_kwargs):
    '''
    Plot 2d PCA 
    The PCA is computed by firstly standardizing the data

    Parameters
    ----------
    ds : pandas dataframe
    
    n_components : int
        number of pca axes to compute 
        
    axes : list or array
        2 pca axes to plot (from 1 to n_components)

    ax : matplotlib.axes._subplots.AxesSubplot, default=None
        add the plot to an existing axis
        
    labels : list of str, default=None
        labels of points if more than one (if one use label instead of labels)
        
    arrows : bool, default=True
        display arrows for feature components (i.e. linear coefficients used in linear regression)
        
    **plt_kwargs

    Returns
    -------
    ax

    '''
    if ax is None:
        ax=plt.gca()
    ds_standard=StandardScaler().fit_transform(ds)
    pca=PCA(n_components=n_components).fit(ds_standard)
    coords=pca.transform(ds_standard)
    pca_ax1=axes[0]-1
    pca_ax2=axes[1]-1
    xpca = coords[:,pca_ax1]
    ypca = coords[:,pca_ax2]
    #ax.scatter(xpca,ypca,**plt_kwargs)
    plotObservations(xpca,ypca,ax=ax,labels=labels,**plt_kwargs)
    
    if arrows:
        xscale=abs(ax.get_xticks()[0]-ax.get_xticks()[1])
        yscale=abs(ax.get_yticks()[0]-ax.get_yticks()[1])
        comp=pca.components_
        feat_names=ds.columns.to_list()
        for i in range(len(comp[pca_ax1])):
            x,y=comp[pca_ax1][i]*xscale,comp[pca_ax2][i]*xscale
            ax.arrow(0,0,x,y,width=width_arrow,fc='k',ec=None,fill=True)
            ax.text(x+width_arrow,y+width_arrow,feat_names[i])
            
        ax.grid(alpha=0.5)
        ax.set_axisbelow(True)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=0, color='k', linestyle='--')

    explained_variance=pca.explained_variance_ratio_
    pca1='Dim'+str(pca_ax1+1)+' ('+str(explained_variance[pca_ax1]*100)[:4]+'%)'
    pca2='Dim'+str(pca_ax2+1)+' ('+str(explained_variance[pca_ax2]*100)[:4]+'%)'
    ax.set(xlabel=pca1,ylabel=pca2)
    
    return(ax)

def plotObservations(x,y,ax=None,labels=None,**plt_kwargs):
    '''
    Plot observation points

    Parameters
    ----------
    x, y : float
        x and y of input observation points
    
    ax : matplotlib.axes._subplots.AxesSubplot, default=None
        add the plot to an existing axis
    
    labels : list of str, default=None
        labels of points if more than one (if one use label instead of labels)
        
    **plt_kwargs

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot

    '''
    if ax is None:
        ax=plt.gca()
    plt_kwargs.setdefault('s',0.1)
    scatter=ax.scatter(x,y,**plt_kwargs)
    if labels is not None:
        handles = scatter.legend_elements(num=len(labels))[0]
        if len(handles)!=len(labels):
            handles = scatter.legend_elements(num=len(labels)-1)[0]
        legend=ax.legend(handles=handles,labels=labels)
        ax.add_artist(legend)
    return(ax)

def plotConfusionMatrix(y_test,y_pred,list_name_classes,ax=None,grid_similar=0,kwargs_sns={},kwargs_plt={}):
    '''
    Plot row normed confusion matrix. Rows are normalized to sum to 100% throughout true classes. The number of observations is also visible on the matrix

    Parameters
    ----------
    y_test  : list or array
        true encoded labels
        
    y_pred : list or array
        predicted encded labels  
        
    list_name_classes : list or array of str
        names of the classified classes. Must be in the same order as encoded labels
        
    grid_similar : int, default=0
        display horizontal and vertical lines to highlight similar classes. This similarity is based on the first grid_similar characters of list_name_classes.
        
    kwargs_sns : dict
    
    kwargs_plt : dict

    Returns
    -------
    matplotlib axis  
    '''
    if ax is None:
        ax=plt.gca()
    kwargs_sns.setdefault('cmap',"YlOrBr")
    cf_matrix = confusion_matrix(y_test, y_pred)
    matrix_percent=[]
    for row in range(cf_matrix.shape[0]):
        matrix_percent.append(cf_matrix[row]/sum(cf_matrix[row]))
    matrix_percent=np.array(matrix_percent)
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in matrix_percent.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(len(list_name_classes),len(list_name_classes))    
    mat = sns.heatmap(matrix_percent, annot=labels,cbar=True, fmt='',vmin=0, vmax=1,ax=ax,
                      cbar_ax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05),**kwargs_sns)
    labels=list_name_classes 
    mat.set_xticklabels(labels)
    mat.set_yticklabels(labels)
    mat.set(ylabel="True Label", xlabel="Predicted Label")
    
    if grid_similar>0:
        count_similar=np.unique(np.array(list(map(lambda x: x[:grid_similar], test))),return_counts=True)[1]
        hlines=vlines=np.cumsum(count_similar)
        createGrid(hlines,vlines,ax)
    
    return(ax)
