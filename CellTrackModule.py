#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:06:32 2018

@author: Oleg Krichevsky okrichev@bgu.ac.il
"""
#import sys

from FitToolModule import FitTool
from FitToolModule import curvefitLims
from FitToolModule import NextItemTool
from FitToolModule import PreviousItemTool
from FitToolModule import WeightedAverage
from FitToolModule import ConfinedDiffusion2DFit

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
#import random
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy import interpolate
from scipy.optimize import newton_krylov
from scipy.optimize import curve_fit
from scipy.special import jn_zeros
from scipy.special import j1 as besselj1
from scipy.signal import medfilt #median filter 
from scipy.signal import savgol_filter # savitsky golay filter
import scipy.io as sio

import pickle
import re



from sklearn.decomposition import PCA
from skimage.feature import peak_local_max

class CellTrackClass:
    def __init__(self):
        self.animalID = -888
        self.movie = ""
        self.cellType = ""
        self.indentifier = np.nan
        self.metaname = ""
        self.CompleteMetaName = ""
        self.behavior = ""
        self.analysis = ""
        self.trackID = np.nan
        self.timePoint = []
        self.t_s = []
        self.X_um = []
        self.Y_um = []
        self.Z_um = []
        self.volume_um3 = []
        self.distance_um = []
        self.surfaceArea_um2 = []
        self.shapeFactor = []
        self.TrackVel_um_h_vol = np.nan
        self.DispVel_um_h_vol = np.nan
        self.MI_vol = np.nan
        self.FitParam = dict()

                        
    def DoSetParams(self, param):
        if isinstance(param, dict):
            for paramName in param:
                setattr(self, paramName, param[paramName])
        else:
            warnings.warn("Param should be a dictionary!")
    
    def DoCalculateVelocities(self):
        #print('Number of points = ' + str(self.t_s.size))
        self.dt_s = np.average(np.diff(self.t_s))
        self.Vx_um_s = np.divide(np.diff(self.X_um),np.diff(self.t_s))
        self.Vy_um_s = np.divide(np.diff(self.Y_um),np.diff(self.t_s))
        self.Vz_um_s = np.divide(np.diff(self.Z_um),np.diff(self.t_s))
        self.V_um_s = np.linalg.norm([self.Vx_um_s, self.Vy_um_s, self.Vz_um_s], axis = 0)
         #minutes
        #minutes        
        self.t_min = (self.t_s)/60
        #vel um/hr
        self.V_um_hr = (self.V_um_s)*3600
        
        self.Vxy_um_s = np.linalg.norm([self.Vx_um_s, self.Vy_um_s], axis = 0)
        self.Vmean_um_s = np.average(self.V_um_s)
        self.Vmean_um_min =  self.Vmean_um_s * 60
        self.Vmeansq_um_s = np.sqrt(np.average(np.square(self.V_um_s)))
        self.Vxymeansq_um_s = np.sqrt(np.average(np.square(self.Vxy_um_s)))
        self.Vxymean_um_s = np.average(self.Vxy_um_s)
         #V(xy)mean um/hr
        self.Vxymean_um_min = (self.Vxymean_um_s)*60
        self.Vdispl_um_s = np.abs(np.linalg.norm([self.X_um[-1]-self.X_um[0], 
                    self.Y_um[-1]-self.Y_um[0], 
                      self.Z_um[-1]-self.Z_um[0]], axis = 0))/self.t_s[-1]
        self.Vxydispl_um_s = np.abs(np.linalg.norm([self.X_um[-1]-self.X_um[0], 
                    self.Y_um[-1]-self.Y_um[0]], axis = 0))/self.t_s[-1]
        self.Vdispl_um_min = self.Vdispl_um_s*60
        self.Vxydispl_um_min = self.Vxydispl_um_s*60
        self.meanderInd = self.Vdispl_um_s/self.Vmean_um_s
        self.meanderXYInd = self.Vxydispl_um_s/self.Vxymean_um_s
   
        self.cosVXYangles = (self.Vx_um_s[1:]*self.Vx_um_s[:-1] + 
                        self.Vy_um_s[1:]*self.Vy_um_s[:-1])/(self.Vxy_um_s[1:]*
                                    self.Vxy_um_s[:-1])
        
    def DoCalculateMSD(self):
        msd = MSDclass()
        msd.Xsq_um2 = np.zeros(self.X_um.size)
        msd.error_Xsq_um2 = np.zeros(self.X_um.size)
        msd.Ysq_um2 = np.zeros(self.X_um.size)
        msd.error_Ysq_um2 = np.zeros(self.X_um.size)
        msd.Zsq_um2 = np.zeros(self.X_um.size)
        msd.error_Zsq_um2 = np.zeros(self.X_um.size)
        
        pca = PCA(n_components=2)
        XY = np.array([self.X_um, self.Y_um]).T
        pca.fit(XY)
        msd.PCA_explained_var = pca.explained_variance_[0]/pca.explained_variance_.sum()
        PCA12 = pca.transform(XY)
        self.Xpca = PCA12[:, 0]
        self.Ypca = PCA12[:, 1]

        msd.Xpca_sq_um2 = np.zeros(self.X_um.size)
        msd.error_Xpca_sq_um2 = np.zeros(self.X_um.size)
        msd.Ypca_sq_um2 = np.zeros(self.X_um.size)
        msd.error_Ypca_sq_um2 = np.zeros(self.X_um.size)
 
        msd.X1X2 = np.zeros(self.X_um.size)
        msd.Y1Y2 = np.zeros(self.X_um.size)
        msd.Z1Z2 = np.zeros(self.X_um.size)
        
        msd.t_s = self.t_s-self.t_s[0]
        Xa, Xb = np.meshgrid(self.X_um, self.X_um)
        Ya, Yb = np.meshgrid(self.Y_um, self.Y_um)
        dist = (Xa - Xb)**2 + (Ya - Yb)**2
        self.maxDist = np.sqrt(dist.max())
        
        for tt in range(1, self.X_um.size):
            diff = self.X_um[tt:] - self.X_um[:-tt]
            msd.Xsq_um2[tt] = np.average(np.square(diff))
            msd.error_Xsq_um2[tt] = stats.sem(np.square(diff))
            msd.X1X2[tt] = np.average(diff[0:-tt]*diff[tt:]) #for confinement determination
            diff = self.Y_um[tt:] - self.Y_um[:-tt]
            msd.Ysq_um2[tt] = np.average(np.square(diff))
            msd.error_Ysq_um2[tt] = stats.sem(np.square(diff))
            msd.Y1Y2[tt] = np.average(diff[0:-tt]*diff[tt:])
            diff = self.Z_um[tt:] - self.Z_um[:-tt]
            msd.Zsq_um2[tt] = np.average(np.square(diff))
            msd.error_Zsq_um2[tt] = stats.sem(np.square(diff))
            msd.Z1Z2[tt] = np.average(diff[0:-tt]*diff[tt:])
            diff = self.Xpca[tt:] - self.Xpca[:-tt]
            msd.Xpca_sq_um2[tt] = np.average(np.square(diff))
            msd.error_Xpca_sq_um2[tt] = stats.sem(np.square(diff))
            diff = self.Ypca[tt:] - self.Ypca[:-tt]
            msd.Ypca_sq_um2[tt] = np.average(np.square(diff))
            msd.error_Ypca_sq_um2[tt] = stats.sem(np.square(diff))

        msd.Rsq_um2 = msd.Xsq_um2 + msd.Ysq_um2 + msd.Zsq_um2
        msd.error_Rsq_um2 = np.sqrt(msd.error_Xsq_um2**2 + msd.error_Ysq_um2**2 +
                         msd.error_Zsq_um2**2)
#        msd.simpleRq = (np.square(self.X_um - self.X_um[0]) +
#                        np.square(self.Y_um - self.Y_um[0]) +
#                        np.square(self.Z_um - self.Z_um[0]))
        msd.XYsq_um2 = msd.Xsq_um2 + msd.Ysq_um2
        msd.error_XYsq_um2 = np.sqrt(msd.error_Xsq_um2**2 + msd.error_Ysq_um2**2)
        #diameter of confinement circle of similar gyration radius
        msd.confinement_XY_um = 4*msd.XYsq_um2/np.sqrt(-4*(msd.X1X2 + msd.Y1Y2)) 
        #estimate confinement from the delays of 1, 2, 3
#        for tt in range(1, 3):
#            X = self.X_um[::tt]
#            dX = np.diff(X)
#            Y = self.Y_um[::tt]
#            dY = np.diff(Y)
#            dX[0:]*dX[:-1]            
        self.MSD = msd
        self.Rg_um = np.sqrt(np.sum(np.var([self.X_um, self.Y_um, self.Z_um], axis = 1)))
      
    def DoTrackWrithe(self):
        # extend last links to infinity
        # what's infinity: estimate trajectory's gyration radius
        self.Rg_um = np.sqrt(np.sum(np.var([self.X_um, self.Y_um, self.Z_um], axis = 1)))
        
        dX = np.diff(self.X_um)
        dY = np.diff(self.Y_um)
        dZ = np.diff(self.Z_um)
        
        if (dX[0] != 0):
            dX[0] = dX[0]/abs(dX[0])*self.Rg_um*100;
         
        if (dY[0] != 0):
            dY[0] = dY[0]/abs(dY[0])*self.Rg_um*100;
            
        if (dZ[0] != 0):
            dZ[0] = dZ[0]/abs(dZ[0])*self.Rg_um*100;

        if (dX[-1] != 0):
            dX[-1] = dX[-1]/abs(dX[-1])*self.Rg_um*100;
            
        if (dY[-1] != 0):
            dY[-1] = dY[-1]/abs(dY[-1])*self.Rg_um*100;
            
        if (dZ[-1] != 0):
            dZ[-1] = dZ[-1]/abs(dZ[-1])*self.Rg_um*100;

        dX = np.insert(dX, 0, 0)
        dY = np.insert(dY, 0, 0)
        dZ = np.insert(dZ, 0, 0)

        X = np.cumsum(dX)
        Y = np.cumsum(dY)
        Z = np.cumsum(dZ)
        
        self.Writhe = DoContourWritheCalc(X, Y, Z)
        
    def DoVelocityCorrelationsOld(self):
        velocityCorr = generalStruc()
        velocityCorr.t_s = self.t_s[:-1]-self.t_s[0]
        velocityCorr.VxVx = np.zeros(velocityCorr.t_s.size)
        velocityCorr.VyVy = np.zeros(velocityCorr.t_s.size)
        velocityCorr.VzVz = np.zeros(velocityCorr.t_s.size)
        velocityCorr.VxyVxy = np.zeros(velocityCorr.t_s.size)
        velocityCorr.VV = np.zeros(velocityCorr.t_s.size)
        for tt in range(1, self.V_um_s.size):
            corr = self.Vx_um_s[tt:] * self.Vx_um_s[:-tt]
            velocityCorr.VxVx[tt] = np.average(corr)
            corr = self.Vy_um_s[tt:] * self.Vy_um_s[:-tt]
            velocityCorr.VyVy[tt] = np.average(corr)
            corr = self.Vz_um_s[tt:] * self.Vz_um_s[:-tt]
            velocityCorr.VzVz[tt] = np.average(corr)
        velocityCorr.VxVx[0] = np.average(np.square(self.Vx_um_s))
        velocityCorr.VyVy[0] = np.average(np.square(self.Vy_um_s))
        velocityCorr.VzVz[0] = np.average(np.square(self.Vz_um_s))
        velocityCorr.VxyVxy = velocityCorr.VxVx + velocityCorr.VyVy
        velocityCorr.VV = velocityCorr.VxyVxy + velocityCorr.VzVz
        velocityCorr.VV = velocityCorr.VV/self.Vmeansq_um_s**2
        velocityCorr.VxyVxy = velocityCorr.VxyVxy/self.Vxymeansq_um_s**2
        self.velocityCorr = velocityCorr
        
    def DoVelocityCorrelations(self):
        velocityCorr = generalStruc()
        velocityCorr.t_s = self.t_s[:-1]-self.t_s[0]
        V = np.array([self.Vx_um_s, self.Vy_um_s, self.Vz_um_s])
        corr = DoCorrelateVectorSeq(V, V)
       # print(corr.shape())
        velocityCorr.VV = corr[-velocityCorr.t_s.size:]/self.Vmeansq_um_s**2
        V = np.array([self.Vx_um_s, self.Vy_um_s])
        corr = DoCorrelateVectorSeq(V, V)
        velocityCorr.VxyVxy = corr[-velocityCorr.t_s.size:]/self.Vxymeansq_um_s**2
        self.velocityCorr = velocityCorr

        
    def DoGetContour(self, coarseRes = [1, 2], XYonly = True):
        #coarseRes = [1, 10, 30]
        self.contour = []
        if XYonly:
            ds = np.linalg.norm([np.diff(self.X_um), np.diff(self.Y_um)], axis = 0)
        else:
            ds = np.linalg.norm([np.diff(self.X_um), np.diff(self.Y_um), np.diff(self.Z_um)], axis = 0)
        ds = np.insert(ds, 0, 0)
        s = np.cumsum(ds)
        cont = generalStruc()
        cont.res_um = 0
        cont.s = s
        cont.XYZ_um = np.array([self.X_um, self.Y_um, self.Z_um])
        cont.L = s[-1]
        #cont.Wr = DoContourWritheCalc(self.X_um, self.Y_um, self.Z_um)
        self.contour.append(cont)
        cont.Rg_um = np.sqrt(self.X_um.var() + self.Y_um.var() + self.Z_um.var())
        
        Fxint = interpolate.interp1d(s, self.X_um)
        Fyint = interpolate.interp1d(s, self.Y_um)
        Fzint = interpolate.interp1d(s, self.Z_um)
        L = cont.L
        
        for res in coarseRes:
            cont1 = generalStruc()
            cont1.res_um = res
            cont1.s = np.arange(0, L, res)
            cont1.XYZ_um = np.array([Fxint(cont1.s), 
                   Fyint(cont1.s), Fzint(cont1.s)])
            cont1.dXYZ_um = np.diff(cont1.XYZ_um, axis = 1)
            if XYonly:
                cont1.L = np.linalg.norm(cont1.dXYZ_um[0:2, :], axis = 0).sum()
            else:
                cont1.L = np.linalg.norm(cont1.dXYZ_um, axis = 0).sum()
            #cont1.L = cont1.s[-1]
            #cont1.Wr = DoContourWritheCalc(cont1.XYZ_um[0, :], 
            #                               cont1.XYZ_um[1, :], cont1.XYZ_um[2, :] )
            #calculate tangent correlations
            cont1.s_tangcorr = cont1.s[:-1]
            if XYonly:
                tangent = cont1.dXYZ_um[0:2, :]
            else:
                tangent = cont1.dXYZ_um
            tangent = tangent/np.linalg.norm(tangent, axis = 0)
            cont1.cosTangent = np.sum(tangent[:,1:] * tangent[:,:-1], 0)
            corr = DoCorrelateVectorSeq(tangent, tangent)
            cont1.tangent = tangent
            cont1.tangent_corr = corr[-cont1.s_tangcorr.size:]
            cont1.Rg_um = np.sqrt(cont1.XYZ_um[:2, :].var(axis=1).sum())
            
#            def BenoitDoty(x): # x has a meaning of x = Lkuhn/L
#                return (cont1.Rg_um/cont1.L)**2 - (x/6)*(1 - 3*x/2 + 
#                       3*x**2/2 - (3*x**3/4)*(1 - np.exp(- 2/x)))
#                
#            cont1.KuhnLength_um = cont1.L*newton_krylov(BenoitDoty, 
#                                                6*(cont1.Rg_um/cont1.L)**2)
            self.contour.append(cont1)
#            if cont1.s.size > 1:
#                Fxint = interpolate.interp1d(cont1.s, cont1.XYZ_um[0, :])
#                Fyint = interpolate.interp1d(cont1.s, cont1.XYZ_um[1, :])
#                Fzint = interpolate.interp1d(cont1.s, cont1.XYZ_um[2, :])
#            L = cont1.L
            
             
    def DoTrackCalculations(self, **kwargs):
        self.DoCalculateVelocities(**kwargs)
        self.DoVelocityCorrelations(**kwargs)
        self.DoCalculateMSD(**kwargs)
#        self.DoTrackWrithe()
        self.DoGetContour(**kwargs)
        
    def DoPlotMSD(self, timeUnits = 'sec', **kwargs):
        if not hasattr(self, 'MSD'):
            self.DoCalculateMSD()
        plt.axis('auto')
        if 'label' not in kwargs:
            if 'labelAttr' not in kwargs:
                kwargs['labelAttr'] = 'CompleteMetaName'
            kwargs['label'] = getattr(self, kwargs['labelAttr'])
            kwargs.pop('labelAttr')
        
        if 'MSDtype' in kwargs:
            msd = getattr(self.MSD, kwargs['MSDtype'])
            Ylabel = kwargs['MSDtype']
            kwargs.pop('MSDtype')
        else:
            msd  = self.MSD.Rsq_um2  
            Ylabel = 'Rsq_um2 '
                    
 #       plt.plot(self.MSD.t_s, self.MSD.Rsq_um2, label = self.CompleteMetaName, **kwargs)
        if timeUnits == 'sec':
            ln, = plt.plot(self.MSD.t_s, msd,  **kwargs)   
            plt.xlabel('t (s)')
        elif timeUnits == 'min':
            ln, = plt.plot(self.MSD.t_s/60, msd,  **kwargs)   
            plt.xlabel('t (min)')
            
        plt.ylabel(Ylabel + ' ($\mu m^2$)')
        plt.legend(loc="best")  
        plt.show()
        
        return ln
        
        def onpick(event):
            if event.artist == ln:
                print(self.CompleteMetaName)

        plt.gcf().canvas.mpl_connect('pick_event', onpick)

        return ln
        
    def DoPlotXYtrack(self, startXYloc = [], markStartEnd = True,  **kwargs):
        if 'label' not in kwargs:
            if 'labelAttr' not in kwargs:
                kwargs['labelAttr'] = 'CompleteMetaName'
            kwargs['label'] = getattr(self, kwargs['labelAttr'])
            kwargs.pop('labelAttr')   

        if 'fontsize' not in kwargs:
            axisLabelFontSize = 18
        else:
            axisLabelFontSize = kwargs['fontsize']
            kwargs.pop('fontsize') 
        
        if 'ShowLegend' not in kwargs:
            ShowLegend = True
        else:
            ShowLegend = kwargs['ShowLegend']
            kwargs.pop('ShowLegend')  

            
        
        if len(startXYloc) == 0:
            XYshift = np.zeros(2)
           # print(startXYloc)
        else:
            XYshift = np.array([self.X_um[0] - startXYloc[0], self.Y_um[0] - startXYloc[1]])
        #print(self.X_um - XYshift[0])  
        #print(self.Y_um - XYshift[1])
        lineHandle, = plt.plot(self.X_um - XYshift[0], self.Y_um - XYshift[1],  **kwargs)   
        
        if markStartEnd:
            col = lineHandle.get_color()
#            if len(startXYloc) == 0:
#                startHandle, = plt.plot(self.X_um[0]- XYshift[0], self.Y_um[0] - XYshift[1], 'o', color = col) 
#            else:
#                startHandle,  = plt.plot(self.X_um[0]- XYshift[0], self.Y_um[0] - XYshift[1], 'o', color = 'k') 
            startHandle, = plt.plot(self.X_um[0]- XYshift[0], self.Y_um[0] - XYshift[1], 'o', ms = 2**2, color = col)
            #l[-1].set_color(col)
            #l = plt.plot(self.X_um[-1], self.Y_um[-1], 'o', fillstyle = 'none')
            plt.arrow(self.X_um[-2] - XYshift[0], self.Y_um[-2] - XYshift[1],
                          self.X_um[-1] - self.X_um[-2], 
                          self.Y_um[-1] - self.Y_um[-2], 
                          color = col, head_width = 2)
        else:
            startHandle = None
           # arrowHandle = None
            
        #l[-1].set_color(col)
        plt.axis('equal')
        #plt.gca().legend([self.CompleteMetaName])
        if ShowLegend:
            plt.legend(loc="best")
        plt.xlabel('X ($\mu m$)', fontsize = axisLabelFontSize)
        plt.ylabel('Y ($\mu m$)', fontsize = axisLabelFontSize)

        plt.show()
        
        def onpick(event):
            if event.artist == lineHandle:
                print(self.CompleteMetaName)

        plt.gcf().canvas.mpl_connect('pick_event', onpick)
        
        return lineHandle, startHandle
        
    def DoPlotInstVel(self, **kwargs):
        if 'label' not in kwargs:
            kwargs['label'] = self.CompleteMetaName
            
        ln, = plt.plot(self.t_s[:-1], self.Vxy_um_s, **kwargs)   
        #col = l[-1].get_color()
        #plt.gca().legend([self.CompleteMetaName])
        plt.legend(loc="best")
        plt.xlabel('t (s)')
        plt.ylabel('VelXY ($\mu m/s$)')
        plt.show()
        
        def onpick(event):
            if event.artist == ln:
                print(self.CompleteMetaName)

        plt.gcf().canvas.mpl_connect('pick_event', onpick)

        
    def DoPlotTrackVel_Vol(self, **kwargs):
        if 'label' not in kwargs:
            kwargs['label'] = self.CompleteMetaName
            
        ln = plt.plot(self.CompleteMetaName, self.TrackVel_um_h_vol, **kwargs)
        plt.legend(loc="best")
        plt.xlabel('mov')
        plt.ylabel('Track Vel ($\mu m/hr$)')
        plt.show()
        
        def onpick(event):
            if event.artist == ln:
                print(self.CompleteMetaName)

        plt.gcf().canvas.mpl_connect('pick_event', onpick)
        
    def DoFitMSD(self, MSDType = 'XYsq_um2', fitfunc = 'LinearFit', 
                 XLim = [-np.inf, np.inf], YLim = [-np.inf, np.inf], ShowXYtrack = False):
        fig = plt.gcf()
        plt.ion()
        if ShowXYtrack:
            ax = plt.subplot(1, 2, 2)
            plt.cla()
            #plt.sca(axs[1])
            self.DoPlotXYtrack(marker = 'o', label = None, fillstyle = 'none')
            TextStr = '\n'.join([
                        r'$expl var=%.2f$' % (self.MSD.PCA_explained_var, ),
                        r'$maxDist =%.2f$' % (self.maxDist, ),
                    #    r'$R_{g, traj}=%.2f$' % (self.contour[1].Rg_um, ),
                    #    r'$L_K =%.2f$' % (self.contour[1].KuhnLength_um, )
                    ])
            ax.text(0.05, 0.05, TextStr, transform=ax.transAxes)
            #plt.sca(axs[0])
            plt.subplot(1, 2, 1)
        
        if (XLim == [-np.inf, np.inf]) and (YLim == [-np.inf, np.inf]): # show and zoom into range
            
            ln = self.DoPlotMSD(MSDtype = MSDType, marker = 'o')
            self.line = ln
            plt.title('Zoom into the fit range and hit Fit button')
            if fitfunc == 'PowerFit':
                plt.xscale('log')
                plt.yscale('log')
            if 'Fit' in fig.canvas.manager.toolmanager.tools:
                fig.canvas.manager.toolmanager.remove_tool('Fit')
            fig.canvas.manager.toolmanager.add_tool('Fit', FitTool, ln, fitfunc, self.MSD, MSDType = MSDType)
            fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Fit'), "toolgroup")
            fig.show()
        else:   
            self.MSD.DoFitMSDLims(fitfunc, XLim = XLim , YLim = YLim, MSDType = MSDType)
            
            
    def DoFindDirectedMotion(self, maxProbability = 0.01, minSteps = 4, Rerror = np.sqrt(0.34**2 + 0.25**2),
                       showPlots = True, RefineProb = False, Dsource = 'none', Dlims = ()): 
        
        self.DirectStretchList = FindDirectedMotion(self, maxProbability, 
                           minSteps, Rerror, showPlots, RefineProb, Dsource, Dlims) # defined down in this file
        
    def DoFindConfinements(self, MinStretchLength = 10, maxProbability = 0.05, Rerror = np.sqrt(0.34**2 + 0.25**2), showPlots = False, **kwargs):
        self.ConfinementList = FindConfinements(self, MinStretchLength = MinStretchLength, 
                         maxProbability = maxProbability, Rerror = Rerror, showPlots = showPlots, **kwargs)
        
    def DoShowTrackStretches(self):
        plt.subplot(1, 1, 1)
        plt.cla()
        self.DoPlotXYtrack(marker = 'o')
        for c in self.DirectStretchList:
            x = c['X']
            y = c['Y']
            P = c['Probabilities']
            #print('P = ' + str(P))
            plt.plot(x, y, marker = 'o', label = 
                    'P = %.2e, V = %.2f $\mu$m/min, L = %.1f $\mu$m, dt = %.1f min '
                     % (P, c['Stretch Speed (um/min)'], c['Stretch Length (um)'], self.dt_s/60))
            plt.legend()
            plt.show()
            
    def DoEstimateDiffusionCoeff(self, MinStretchLength = 10):
        D, confinementRadius, D1, msd, MSDerror, t, t1, msd1, intercept = EstimateDiffusionCoeff(X = self.X_um, Y = self.Y_um,
                                               t = self.t_s, 
                                               DirectStretchList = self.DirectStretchList, 
                                               MinStretchLength = MinStretchLength)
        
        plt.subplot(1, 2, 1)
        plt.cla()
        self.DoPlotXYtrack(marker = 'o')
        for c in self.DirectStretchList:
            x = c['X']
            y = c['Y']
            P = c['Probabilities']
            #print('P = ' + str(P))
            plt.plot(x, y, marker = 'o', label = 
                    'P = %.2e, V = %.2f $\mu$m/min, L = %.1f $\mu$m, dt = %.1f min '
                     % (P, c['Stretch Speed (um/min)'], c['Stretch Length (um)'], self.dt_s/60))
 #           plt.legend()
            plt.show()
        plt.subplot(1, 2, 2)
        # do fit

        plt.cla()
        plt.plot(t, msd, '-o')
        plt.plot(t1, intercept + 4*D*t1, label = 'P = %.2e' % (D))
        plt.plot(t1, ConfinedDiffusion2DFit(t1, confinementRadius**2, 
                                            4*D/confinementRadius**2)) 
        plt.legend()
        plt.show()
        
    def DoShowTrackDirectedAndConfined(self, ClearPrevious = True, startXYloc = [], ShowLegend = True):
        if len(startXYloc) == 0:
            XYshift = np.zeros(2)
           # print(startXYloc)
        else:
            XYshift = np.array([self.X_um[0] - startXYloc[0], self.Y_um[0] - startXYloc[1]])

        plt.subplot(1, 1, 1)
        if ClearPrevious:
            plt.cla()
        self.DoPlotXYtrack(linestyle='dashed', startXYloc = startXYloc, ShowLegend = ShowLegend)
        for c in self.DirectStretchList:
            x = c['X'] - XYshift[0]
            y = c['Y'] - XYshift[1]
            P = c['Probabilities']
            #print('P = ' + str(P))
            plt.plot(x, y, marker = 'o', ms = 2**2, label = 
                    'P = %.2e, V = %.2f $\mu$m/min, L = %.1f $\mu$m, dt = %.1f min '
                     % (P, c['Stretch Speed (um/min)'], c['Stretch Length (um)'], self.dt_s/60))
            
        for confinement in self.ConfinementList:
            P = confinement['Probabilities']
            print('P = ' + str(P))
            x = confinement['X'] - XYshift[0]
            y = confinement['Y'] - XYshift[1]
            plt.plot(x, y, marker = 'o', ms = 2**2, mfc='none', color = 'k', label = 
                    'P = %.2e, Npoints = %d, Size (um) = %.2e'
                     % (P, x.size, confinement['Max Displacement (um)']))
        if ShowLegend:
            plt.legend()
        plt.show()
        
    def DoSaveDataToMatlabFile(self, folderName, version = 1, **kwargs):
        if folderName[-1] != '/':
            folderName = folderName + '/'
        if version == 0:    
            fname = 'CellCoordImport'
            sio.savemat(folderName+fname, 
                     {'StrucName': fname, 
                      'X_um' : self.X_um, 'Y_um' : self.Y_um, 'Z_um' : self.Z_um})
            fname = 'ROI'
            sio.savemat(folderName+fname, 
                     {'StrucName': fname, 
                      'x0' : self.ROI.x0, 'y0' : self.ROI.y0, 'z0' : self.ROI.z0,
                      'x1' : self.ROI.x1, 'y1' : self.ROI.y1, 'z1' : self.ROI.z1})
            
            for idx, DS in enumerate(self.DirectStretchList, start = 1):
                fname = 'DirectedStretch'
                sio.savemat(folderName+fname+str(idx), 
                     {'StrucName': fname, 'index' : idx,
                      'X' : DS['X'], 'Y' : DS['Y']})
        else: #version = 1
            if 'filename' in kwargs:
                fname = kwargs['filename']
            else:
                fname = 'PythonDataImport'
                
            propNames = ['CompleteMetaName', 't_s', 'dt_s', 'X_um', 'Y_um', 'Z_um', 
                         'ROI', 'DirectStretchList', 'ConfinementList']
            if 'propNames' in kwargs:
                propNames.extend(kwargs['propNames'])
            
            # make property names valid Matlab names
            #clean = lambda varStr: re.sub('\W|^(?=\d)','_', varStr)
            
            saveDict = {'StrucName': fname, 'version' : version}
            for pN in propNames:
                print(pN)
                saveDict[pN] = getattr(self, pN)
            print(folderName+fname)
            sio.savemat(folderName+fname, saveDict)
            

            
"""
CellTrackArrayClass
"""
            
class CellTrackArrayClass:
    def __init__(self):
        self.tracks = []
        self.metanameList = []
        self.SaveFileName = ''
        
    def DoLoadDataFromExcelFile(self, ExcelFilePath, FiltFunc = None, FiltWindow = 3, FiltOrder = 3):
        D = pd.read_excel(ExcelFilePath)
        for AnmlID in D['Animal ID'].unique():
            #print(AnmlID)
            singleAnimal = (D.loc[D['Animal ID'] == AnmlID])
            for mov in singleAnimal['movie'].unique():
               # print(mov)
                singleMovie = (singleAnimal.loc[singleAnimal['movie'] == mov])
                for metaNm in singleMovie['metaname'].unique():
                    singleTrack = (singleMovie.loc[singleMovie['metaname'] == metaNm]) 
                    if singleTrack["cell type"].unique().size > 1:
                        warnings.warn("Cell type is not unique in " + metaNm + " !")
                        
                    newTrack = CellTrackClass()
                    newTrack.DoSetParams({"animalID" : int(AnmlID), "movie" : mov, "metaname" : metaNm,
                                          "X_um" :  singleTrack["Centroid X (µm)"].values,
                                          "Y_um" :  singleTrack["Centroid Y (µm)"].values,
                                          "Z_um" :  singleTrack["Centroid Z (µm)"].values, 
                                          "t_s" : singleTrack["Rel. Time (s)"].values,
                                          "cellType" : singleTrack["cell type"].unique(),
                                          "trackID" : singleTrack["Track ID"].unique(),
                                          "volume_um3" : singleTrack["Volume (µm³)"].values,
                                          "surfaceArea_um2" : singleTrack["Surface Area (µm²)"].values,
                                          "shapeFactor" : singleTrack["Shape Factor"].values,
                                           "TrackVel_um_h_vol": singleTrack["Track Velocity (µm/hr)"].values,
                                          "DispVel_um_h_vol": singleTrack["Displacement Rate (µm/hr)"].values,
                                          "MI_vol": singleTrack["Meandering Index"].values})
                    ROI = generalStruc()
                    ROI.x0 = singleTrack["ROIx_start"].unique()
                    ROI.x1 = singleTrack["ROIx_end"].unique()
                    ROI.y0 = singleTrack["ROIy_start"].unique()
                    ROI.y1 = singleTrack["ROIy_end"].unique()
                    ROI.z0 = singleTrack["ROIz_start"].unique()
                    ROI.z1 = singleTrack["ROIz_end"].unique()

                    newTrack.ROI = ROI
                    newTrack.CompleteMetaName = str(newTrack.animalID) + '_' + newTrack.metaname
                    newTrack.duration_h = (newTrack.t_s[-1] - newTrack.t_s[0])/3600
                    newTrack.duration_h_legend = str(np.round(newTrack.duration_h, decimals = 1)) + 'h'
                    if (FiltFunc == medfilt):
                        newTrack.X_um = FiltFunc(newTrack.X_um, FiltWindow)
                        newTrack.Y_um = FiltFunc(newTrack.Y_um, FiltWindow)
                        newTrack.Z_um = FiltFunc(newTrack.Z_um, FiltWindow)
                        
                    if (FiltFunc == savgol_filter):
                        if newTrack.X_um.size >= FiltWindow:
                            newTrack.X_um = FiltFunc(newTrack.X_um, FiltWindow, FiltOrder)
                            newTrack.Y_um = FiltFunc(newTrack.Y_um, FiltWindow, FiltOrder)
                            newTrack.Z_um = FiltFunc(newTrack.Z_um, FiltWindow, FiltOrder)
   
                    newTrack.DoCalculateVelocities()
                    self.tracks.append(newTrack)
                    self.metanameList.append(newTrack.CompleteMetaName)
 
    def DoLoadZachsDataFromExcelFile(self, ExcelFilePath):
        D = pd.read_excel(ExcelFilePath)
        for TrackID in D['TrackID'].unique():
            #print(AnmlID)
            singleTrack = (D.loc[D['TrackID'] == TrackID])
            newTrack = CellTrackClass()
            newTrack.DoSetParams({"animalID" : int(TrackID),  "metaname" : str(TrackID),
                                  "X_um" :  singleTrack["Position X"].values,
                                  "Y_um" :  singleTrack["Position Y"].values,
                                  "Z_um" :  singleTrack["Position Z"].values, 
                                  "t_s" : singleTrack["Time"].values,
                                  "trackID" : int(TrackID)})
            
            newTrack.CompleteMetaName = str(newTrack.animalID)
            newTrack.t_s = newTrack.t_s - newTrack.t_s[0]
            newTrack.DoCalculateVelocities()
            self.tracks.append(newTrack)
            self.metanameList.append(newTrack.CompleteMetaName)

    
    def DoGetCellTrackSubarrayByParamValue(self, param, operation = 'equal'):
        SubArray = CellTrackArrayClass()
        if isinstance(param, dict):
            selectedTracks = self.tracks
            for paramName in param:
                if operation == 'equal':
                    selectedTracks = [trk for trk in selectedTracks if 
                                  getattr(trk, paramName) == param[paramName]]
                elif operation == 'greater':
                    selectedTracks = [trk for trk in selectedTracks if 
                                  getattr(trk, paramName) > param[paramName]]
                elif operation == 'lesser':
                    selectedTracks = [trk for trk in selectedTracks if 
                                  getattr(trk, paramName) < param[paramName]]

            SubArray.tracks= selectedTracks
        else:
            warnings.warn("Param should be a dictionary!")
        SubArray.DoMetanameList() 
        
        return SubArray

    def DoGetCellTrackSubarrayBySubstring(self, param):
        def IsAnySubstringInListInaString(ListOfSubStrings, aString):
            if isinstance(ListOfSubStrings, list):
                isthere = [(aString.find(substr)!= -1) for substr in ListOfSubStrings]
                return any(isthere)
            else: #assumed it is just a single string
                return (aString.find(ListOfSubStrings) != -1)
                    
        SubArray = CellTrackArrayClass()
        if isinstance(param, dict):
            selectedTracks = self.tracks
            for paramName in param:
#                selectedTracks = [trk for trk in selectedTracks if 
#                                  (getattr(trk, paramName).find(param[paramName]) != -1)]

                selectedTracks = [trk for trk in selectedTracks if 
                                  IsAnySubstringInListInaString(param[paramName], 
                                                                getattr(trk, paramName))]
                
            SubArray.tracks= selectedTracks
        else:
            warnings.warn("Param should be a dictionary!")
        SubArray.DoMetanameList() 
        
        return SubArray  

    def DoExcludeCellTracksBySubstring(self, param):
        SubArray = CellTrackArrayClass()
        if isinstance(param, dict):
            selectedTracks = self.tracks
            for paramName in param:
                selectedTracks = [trk for trk in selectedTracks if 
                                  (getattr(trk, paramName).find(param[paramName]) == -1)] 
            SubArray.tracks= selectedTracks
        else:
            warnings.warn("Param should be a dictionary!")
        SubArray.DoMetanameList()  
        
        return SubArray           
               
    def DoCalculateMSD(self):
        for trk in self.tracks :
            trk.DoCalculateMSD()
            
    def DoTrackCalculations(self):
        for trk in self.tracks :
            trk.DoTrackCalculations()
            
    def DoSortTracksByParam(self, paramName):
        tracksSorted = sorted(self.tracks, key=lambda trk: getattr(trk, paramName))
        self.tracks =  tracksSorted
        
    def DoGetTrackParamByName(self, paramName):
        return [getattr(trk, paramName) for trk in self.tracks]
    
    def DoFindMatches(self, otherTracks, paramNameList):
        matchesDict = {}
        for trk in self.tracks:
            metaName = trk.CompleteMetaName
            matchingTracks = otherTracks
            for paramName in paramNameList:
                paramVal = getattr(trk, paramName)
                if paramName != 'movie':                    
                    matchingTracks = matchingTracks.DoGetCellTrackSubarrayByParamValue({paramName : paramVal})
                else: #same movie can be labeled slightly differently: first substring before dash is the movie name
                    paramVal = paramVal.split('-')[0]
                    matchingTracks = matchingTracks.DoGetCellTrackSubarrayBySubstring({paramName : paramVal})
            matchesDict[metaName] = matchingTracks
        return matchesDict
    
    def DoTrackDict(self):
        trackDict = {}
        for trk in self.tracks:
            trackDict[trk.CompleteMetaName] = trk
        return trackDict
    
    def DoMetanameList(self):
        self.metanameList = [trk.CompleteMetaName for trk in self.tracks]
    
    def DoPlotMSD(self, timeUnits = 'sec', **kwargs):
#       ln2meta = dict()
        for trk in self.tracks:
            trk.DoPlotMSD(timeUnits, **kwargs)
#            ln = trk.DoPlotMSD(**kwargs)
#            ln.set_picker(5)
#            ln2meta[ln] = trk.CompleteMetaName
#
#        def onpick(event):
#            ln = event.artist
#            print(ln2meta[ln])
#
#        plt.gcf().canvas.mpl_connect('pick_event', onpick)
            
            
    def DoPlotXYtrack(self, **kwargs):
        for trk in self.tracks:
            trk.DoPlotXYtrack(**kwargs)
        
    def DoPlotInstVel(self, **kwargs):
        for trk in self.tracks:
            trk.DoPlotInstVel(**kwargs)
            
    def DoPlotTrackVel_Vol(self, **kwargs):
        for trk in self.tracks:
            trk.DoPlotTrackVel_Vol(**kwargs)
            
    def DoFitMSDLims(self, MSDType = 'XYsq_um2', fitName = 'LinearFit', 
                 XLim = [-np.inf, np.inf], YLim = [-np.inf, np.inf]):
        for trk in self.tracks:
            trk.MSD.DoFitMSDLims(self, MSDType = 'XYsq_um2', fitName = 'LinearFit', XLim = XLim, YLim = YLim)
            
    def DoFitMSD(self, MSDType = 'XYsq_um2', fitfunc = 'LinearFit', 
                 XLim = [-np.inf, np.inf], YLim = [-np.inf, np.inf], ShowXYtrack = False):
        if (XLim == [-np.inf, np.inf]) and (YLim == [-np.inf, np.inf]): # show and zoom into range
            fig = plt.gcf()
            plt.ion()
            if 'Previous Item' in fig.canvas.manager.toolmanager.tools:
                fig.canvas.manager.toolmanager.remove_tool('Previous Item')
            fig.canvas.manager.toolmanager.add_tool('Previous Item', PreviousItemTool,
                        self.tracks, 'DoFitMSD', fitfunc = fitfunc, MSDType = MSDType, ShowXYtrack = ShowXYtrack)
            fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Previous Item'), "toolgroup")

            
            if 'Next Item' in fig.canvas.manager.toolmanager.tools:
                fig.canvas.manager.toolmanager.remove_tool('Next Item')
            fig.canvas.manager.toolmanager.add_tool('Next Item', NextItemTool,
                        self.tracks, 'DoFitMSD', fitfunc = fitfunc, MSDType = MSDType, ShowXYtrack = ShowXYtrack)
            fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Next Item'), "toolgroup")
            fig.show()
        else:   
            self.MSD.DoFitMSDLims(fitfunc, XLim = XLim , YLim = YLim, MSDType = MSDType)

    def DoFindDirectedMotion(self, maxProbability = 0.01, minSteps = 4, Rerror = np.sqrt(0.34**2 + 0.25**2),
                       showPlots = False, RefineProb = False, Dsource = 'none', Dlims = ()):   
        for trk in self.tracks:
            trk.DoFindDirectedMotion(maxProbability, minSteps, Rerror,
                       showPlots, RefineProb, Dsource, Dlims)
    
    def DoFindConfinements(self, MinStretchLength = 10, maxProbability = 0.05, Rerror = np.sqrt(0.34**2 + 0.25**2), 
                           showPlots = False, **kwargs):   
        for trk in self.tracks:
            print(trk.CompleteMetaName)
            trk.DoFindConfinements(MinStretchLength, maxProbability, Rerror, showPlots, **kwargs)
        
    def DoShowConfinements(self, MinStretchLength = 10, maxProbability = 0.05, Rerror = np.sqrt(0.34**2 + 0.25**2), 
                           showPlots = False, **kwargs):
        if showPlots:
            fig = plt.gcf()
            plt.ion()
            if 'Previous Item' in fig.canvas.manager.toolmanager.tools:
                fig.canvas.manager.toolmanager.remove_tool('Previous Item')
            fig.canvas.manager.toolmanager.add_tool('Previous Item', PreviousItemTool,
                        self.tracks, 'DoFindConfinements', MinStretchLength = MinStretchLength, 
                        maxProbability = maxProbability, Rerror = Rerror, showPlots = showPlots, **kwargs)
            fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Previous Item'), "toolgroup")
    
            
            if 'Next Item' in fig.canvas.manager.toolmanager.tools:
                fig.canvas.manager.toolmanager.remove_tool('Next Item')
            fig.canvas.manager.toolmanager.add_tool('Next Item', NextItemTool,
                        self.tracks, 'DoFindConfinements', MinStretchLength = MinStretchLength, 
                        maxProbability = maxProbability, Rerror = Rerror, showPlots = showPlots, **kwargs)
            fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Next Item'), "toolgroup")
            fig.show()
        else:    
            for trk in self.tracks:
                trk.DoFindConfinements(MinStretchLength, maxProbability, Rerror, showPlots)
            
    def DoShowTrackStretches(self):
        fig = plt.gcf()
        plt.ion()
        if 'Previous Item' in fig.canvas.manager.toolmanager.tools:
            fig.canvas.manager.toolmanager.remove_tool('Previous Item')
        fig.canvas.manager.toolmanager.add_tool('Previous Item', PreviousItemTool,
                    self.tracks, 'DoShowTrackStretches')
        fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Previous Item'), "toolgroup")

        
        if 'Next Item' in fig.canvas.manager.toolmanager.tools:
            fig.canvas.manager.toolmanager.remove_tool('Next Item')
        fig.canvas.manager.toolmanager.add_tool('Next Item', NextItemTool,
                    self.tracks, 'DoShowTrackStretches')
        fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Next Item'), "toolgroup")
        fig.show()
        
    def DoShowTrackDirectedAndConfined(self):
        fig = plt.gcf()
        plt.ion()
        if 'Previous Item' in fig.canvas.manager.toolmanager.tools:
            fig.canvas.manager.toolmanager.remove_tool('Previous Item')
        fig.canvas.manager.toolmanager.add_tool('Previous Item', PreviousItemTool,
                    self.tracks, 'DoShowTrackDirectedAndConfined')
        fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Previous Item'), "toolgroup")

        
        if 'Next Item' in fig.canvas.manager.toolmanager.tools:
            fig.canvas.manager.toolmanager.remove_tool('Next Item')
        fig.canvas.manager.toolmanager.add_tool('Next Item', NextItemTool,
                    self.tracks, 'DoShowTrackDirectedAndConfined')
        fig.canvas.manager.toolbar.add_tool(fig.canvas.manager.toolmanager.get_tool('Next Item'), "toolgroup")
        fig.show()
        
#    def save(self, *args):
#        if len(args) > 0:
#            filename = args[0]
#        else:
#            filename = self.SaveFileName
#            
#        if len(filename) < 1:
#            print('Provide the filename for saving!')
#            return
#        self.SaveFileName = filename
#        
#        with open(filename, 'wb') as output:
#            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)       
#            output.close()
            
        
        
    @classmethod
    def loader(cls, filename):
        with open(filename, 'rb') as input:
            obj =  pickle.load(input)
            #obj.__dict__.update(tempDict)
            return obj



         
            
class CellTrackMatchClass:
    def __init__(self, keyObj):
        self.keyDict = keyObj.DoTrackDict()
        self.matchDict = {}
        
#    def DoMatchTracks(self, keyObj, matchingObj, paramNameList, matchingDictName):
    def DoMatchTracks(self, matchingObj, paramNameList, matchingDictName):
     #   self.keyDict = keyObj.DoTrackDict()
 #       self.matchDict[matchingDictName] = keyObj.DoFindMatches(matchingObj, paramNameList)
        matchesDict = {}
        for metaname, trk in self.keyDict.items():
 #           metaName = trk.CompleteMetaName
            matchingTracks = matchingObj
            for paramName in paramNameList:
                paramVal = getattr(trk, paramName)
                if paramName != 'movie':                    
                    matchingTracks = matchingTracks.DoGetCellTrackSubarrayByParamValue({paramName : paramVal})
                else: #same movie can be labeled slightly differently: first substring before dash is the movie name
                    paramVal = paramVal.split('-')[0]
                    matchingTracks = matchingTracks.DoGetCellTrackSubarrayBySubstring({paramName : paramVal})
            matchesDict[metaname] = matchingTracks
        self.matchDict[matchingDictName] = matchesDict

    
    def DoPlotMSD(self, matchDictName, metaname, **kwargs):
        self.keyDict[metaname].DoPlotMSD()
        v = self.matchDict[matchDictName][metaname]
        if len(v.tracks) > 0:
            v.DoPlotMSD(**kwargs)
        else:
            print('No matching ' + matchDictName + ' for ' + metaname + ' !')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.subplots_adjust(right= 0.7)

#        for key, v in self.matchDict[matchDictName].items():
#            if len(v[metaname].tracks) > 0:
#                v[metaname].DoPlotMSD()
#            else:
#                print('No matching ' + key + ' for ' + metaname + ' !')
                
    def DoPlotXYtrack(self, matchDictName, metaname, **kwargs):
        self.keyDict[metaname].DoPlotXYtrack()
        v = self.matchDict[matchDictName][metaname]
        if len(v.tracks) > 0:
            v.DoPlotXYtrack(**kwargs)
        else:
            print('No matching ' + matchDictName + ' for ' + metaname + ' !')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.subplots_adjust(right= 0.7)

#        for key, v in self.matchDict[matchDictName].items():
#            if len(v[metaname].tracks) > 0:
#                v[metaname].DoPlotXYtrack()
#            else:
#                print('No matching ' + key + ' for ' + metaname + ' !')
    
    def DoPlotInstVel(self, matchDictName, metaname, **kwargs):
        self.keyDict[metaname].DoPlotInstVel()
        v = self.matchDict[matchDictName][metaname]
        if len(v.tracks) > 0:
            v.DoPlotInstVel(**kwargs)
        else:
            print('No matching ' + matchDictName + ' for ' + metaname + ' !')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.subplots_adjust(right= 0.7)      
        
        
class MSDclass:
    def __init__(self):
        self.FitParam = dict()
    
#    pass
    def DoFitMSDLims(self, MSDType = 'XYsq_um2', fitName = 'LinearFit', 
                 XLim = [-np.inf, np.inf], YLim = [-np.inf, np.inf]):
        print(MSDType)
        msd = getattr(self, MSDType)
        error_msd = getattr(self, 'error_' + MSDType)
        FP = curvefitLims(fitName, self.t_s, msd, error_msd, XLim, YLim)
        print('beta = ' + str(FP['beta']))
        print('chiSqNorm = ' + str(FP['chiSqNorm']))
        FP['MSDType'] = MSDType
        self.FitParam[fitName] = FP

         
class generalStruc:
    pass
   
"""
Helper functions: not related to class
"""             

def DoGetOmegaForWritheCalc(R0123):
    Rij = np.array([R0123[2]-R0123[0], R0123[3]-R0123[0], 
           R0123[3]-R0123[1], R0123[2]-R0123[1]])
    Rij_rolled = np.roll(Rij, -1, axis = 0)
    
    n0123 = np.cross(Rij, Rij_rolled)
    
    normvec = np.linalg.norm(n0123, axis = 1)
    
    n0123norm = n0123/normvec[:, None]
    n0123norm[n0123 == 0] = 0 # take care of division by 0
    
 #   test_norm = np.linalg.norm(n0123norm, axis = 1)
    
#    OmegaStar = (np.arcsin(np.dot(n0123norm[0], n0123norm[1])) + 
#                 np.arcsin(np.dot(n0123norm[1], n0123norm[2])) +
#                 np.arcsin(np.dot(n0123norm[2], n0123norm[3])) + 
#                 np.arcsin(np.dot(n0123norm[3], n0123norm[0])))
    
    cosarray = np.array([np.dot(n0123norm[0], n0123norm[1]),
                         np.dot(n0123norm[1], n0123norm[2]),
                         np.dot(n0123norm[2], n0123norm[3]),
                         np.dot(n0123norm[3], n0123norm[0])])
    cosarray[cosarray > 1] = 1
    cosarray[cosarray < -1] = -1
    OmegaStar = np.sum(np.arcsin(cosarray))
    R23 = np.array(R0123[3]-R0123[2])
    R01 = np.array(R0123[1]-R0123[0])
    Omega = OmegaStar*np.sign(np.dot(np.cross(R23, R01), Rij[0]))
    
    return Omega

def DoContourWritheCalc(X, Y, Z):
    Wr = 0
    N = X.shape[0]
    
    for segI in range(2, N):
        for segJ in range(1, segI-1):
            if ((segI == N-1) and (segJ == 1)):
                continue
            R0123 = np.array([[X[segI-1], Y[segI-1], Z[segI-1]], [X[segI], Y[segI], Z[segI]],
                     [X[segJ-1], Y[segJ-1], Z[segJ-1]], [X[segJ], Y[segJ], Z[segJ]]])
            Omega = DoGetOmegaForWritheCalc(R0123)
 #           print(segI, segJ, Omega)
            Wr = Wr + Omega
    Wr = Wr/2/np.pi
    
    return Wr

def DoCorrelateVectorSeq(V, U):
   # vector components (say XYZ) are in different rows, and time sequence 
   # is across the columns say Vx(t) = V[0, :]
   
   #multiplicaton matrix: zero time delay is the main diagonal other time dlays
   # on other diagonals
   MM = np.matmul(np.transpose(V), U)
   
   #arrange the matrix of time delays: cannot use negatives for bincount
   # zero index is at top right corner
   rows, cols = MM.shape
   rows_arr = np.arange(rows)
   cols_arr = np.arange(cols)
   diag_idx = rows_arr[:, None] - (cols_arr - (cols - 1))
   corr = np.bincount(diag_idx.ravel(), weights = MM.ravel())/np.bincount(diag_idx.ravel())
   return corr

def DoCalculateMSD(X, Y = np.empty((1, 0)), Z = np.empty((1, 0)), t = np.empty((1, 0))):
    # assumes
    msd = MSDclass()
    msd.Xsq_um2 = np.zeros(X.size)
    msd.error_Xsq_um2 = np.zeros(X.size)
    msd.X1X2 = np.zeros(X.size)
    if Y.size == 0:
        msd.Ysq_um2 = np.empty((1, 0))
        msd.error_Ysq_um2 = np.empty((1, 0))
        msd.Y1Y2 = np.empty((1, 0))
    else:
        msd.Ysq_um2 = np.zeros(Y.size)
        msd.error_Ysq_um2 = np.zeros(Y.size)
        msd.Y1Y2 = np.zeros(Y.size)

    if Z.size == 0:
        msd.Zsq_um2 = np.empty((1, 0))
        msd.error_Zsq_um2 = np.empty((1, 0))
        msd.Z1Z2 = np.empty((1, 0))
    else:          
        msd.Zsq_um2 = np.zeros(Z.size)
        msd.error_Zsq_um2 = np.zeros(Z.size)
        msd.Z1Z2 = np.zeros(Z.size)
    
    if t.size != 0:    
        msd.t_s = t - t[0]
    else:
        msd.t_s = np.arange(X.size)
        
    Xa, Xb = np.meshgrid(X, X)
    Xab = (Xa - Xb)
    dist = Xab**2
    if Y.size != 0:
        Ya, Yb = np.meshgrid(Y, Y)
        Yab = (Ya - Yb)
        dist = dist + Yab**2
        
    msd.maxDist = np.sqrt(dist.max())
    
    if Z.size != 0:
        Za, Zb = np.meshgrid(Z, Z)
        Zab = (Za - Zb)
#        dist = dist + Zab**2

    
    for tt in range(1, X.size):
 #       print('tt = ' + str(tt))
        diff = Xab.diagonal(offset = tt)
        msd.Xsq_um2[tt] = np.average(np.square(diff))
        msd.error_Xsq_um2[tt] = stats.sem(np.square(diff))
        msd.X1X2[tt] = np.average(diff[:-tt]*diff[tt:]) #for confinement determination
        
        if Y.size != 0:
            diff = Yab.diagonal(offset = tt)
            msd.Ysq_um2[tt] = np.average(np.square(diff))
            msd.error_Ysq_um2[tt] = stats.sem(np.square(diff))
            msd.Y1Y2[tt] = np.average(diff[:-tt]*diff[tt:])
        
        if Z.size != 0:
            diff = Zab.diagonal(offset = tt)
            msd.Zsq_um2[tt] = np.average(np.square(diff))
            msd.error_Zsq_um2[tt] = stats.sem(np.square(diff))
            msd.Z1Z2[tt] = np.average(diff[:-tt]*diff[tt:])
    
    if (Z.size*Y.size) != 0:        
        msd.Rsq_um2 = msd.Xsq_um2 + msd.Ysq_um2 + msd.Zsq_um2
        msd.error_Rsq_um2 = np.sqrt(msd.error_Xsq_um2**2 + msd.error_Ysq_um2**2 +
                         msd.error_Zsq_um2**2)
    if Y.size != 0:
        msd.XYsq_um2 = msd.Xsq_um2 + msd.Ysq_um2
        msd.error_XYsq_um2 = np.sqrt(msd.error_Xsq_um2**2 + msd.error_Ysq_um2**2)
        #diameter of confinement circle of similar gyration radius
        msd.confinement_XY_um = 4*msd.XYsq_um2/np.sqrt(-4*(msd.X1X2 + msd.Y1Y2)) 
                
    return msd


def intervalOverlap(interval1, intervalArray): # finds whether there is
# an overlap between interval1 and the intervals in the array: is used in FindDirected motion
    center = interval1.mean()
    rad = np.abs(np.diff(interval1)[0])/2
    centers = intervalArray.mean(axis =1)
    rads = np.abs(np.diff(intervalArray, axis = 1)[:, 0])/2
    return np.abs(centers - center) < (rad + rads)


def FindDirectedMotion(trk, maxProbability = 0.1, minSteps = 2, Rerror = np.sqrt(0.34**2 + 0.25**2),
                       showPlots = True, RefineProb = False, Dsource = 'none', Dlims = ()):
    DirectStretchList = []
    Llim = minSteps
    dr2_error = Rerror**2
    x = trk.X_um
    y = trk.Y_um
    X1, X2 = np.meshgrid(x, x)
    Y1, Y2 = np.meshgrid(y, y)
    Rsq = (X1 - X2)**2 + (Y1 - Y2)**2
    I, J = np.meshgrid(np.arange(x.size), np.arange(x.size))
    
    diffX = np.diff(x)
    diffY = np.diff(y)
    ds2 = diffX**2 + diffY**2
    # taking the average of ds on the stretch itself
    #cumsumDs2 = np.cumsum(np.insert(ds2, 0, 0))
    #CS1, CS2 = np.meshgrid(cumsumDs2, cumsumDs2)
    #s2 = np.abs(CS1 - CS2)
    #rat = Rsq/s2
    Dlims = np.array(Dlims)/60
    if Dsource == 'none':
        D = (ds2.mean()-dr2_error)/(4*trk.dt_s)
    elif Dsource == 'confinements':
       # print('here')
        if hasattr(trk, 'ConfinementList'):
            if len(trk.ConfinementList) > 0 :
                D = trk.ConfinementList[0]['Average Diff Coeff (um^2/min)']/60                 
                if Dlims.size == 1: # e.g Dlims = (1, )
                    D = Dlims[0]
                elif Dlims.size == 3: #e.g. Dlims = (0.2, 1, 4)  in um^2/min !!
                    if np.isnan(D):
                        D = Dlims[1]
                    elif (D < Dlims[0]):
                        D = Dlims[0]
                    elif (D > Dlims[2]):
                        D = Dlims[2]
            else:
                if Dlims.size == 1: # e.g Dlims = (1, )
                    D = Dlims[0]
                elif Dlims.size == 3: #e.g. Dlims = (0.2, 1, 4)  in um^2/min !!
                    D = Dlims[1]
        else:
            if Dlims.size == 1: # e.g Dlims = (1, )
                D = Dlims[0]
            elif Dlims.size == 3: #e.g. Dlims = (0.2, 1, 4)  in um^2/min !!
                D = Dlims[1]


    print('D = ', D*60)
                
    rat = (Rsq - Rerror**2)/(4*D*trk.dt_s*np.abs(I-J))
    logP = -rat +  np.log((x.size-1)/(np.abs(I-J)-1)) #np.log(x.size - np.abs(I-J)) # # - 4*np.abs(I-J)/trk.X_um.size
    
    for tt in range(1, Llim):
        logP[range(x.size - tt), range(tt, x.size)] = np.log(3*maxProbability)
        logP[range(tt, x.size),  range(x.size - tt)] = np.log(3*maxProbability)

    if showPlots:
        plt.cla()
        plt.subplot(1, 2, 1)
        plt.imshow(-logP)
        plt.subplot(1, 2, 2)
        plt.cla()
        trk.DoPlotXYtrack(marker = 'o')

        
        
    logP[np.isnan(logP)] = logP.max()
    temp = -logP
    coord = peak_local_max(temp, threshold_abs = -np.log(3*maxProbability), exclude_border = False)
    if coord.size == 0:
        return DirectStretchList
    #print(logP[coord[:, 0], coord[:, 1]])
    #print(coord.size)
    ii = np.argsort(-temp[coord[:, 0], coord[:, 1]]) #- sign to have a descending sort
    ii = ii[::2] # remove same peaks from the other matrix triangle
#    print(coord)
    coord = coord[ii, :]
#    print(coord)
    coord.sort(axis = 1) #make sure that step indexes in the move are arrange in ascending order
#    print(coord)
    #dismiss short stretches
    coord = coord[(np.diff(coord, axis =1) >= Llim).T[0]]
    if coord.size == 0:
        plt.show()
        return DirectStretchList
    #print(np.exp(logP[coord[:, 0], coord[:, 1]]))
     
    # remove overlaping intervals 
    coord2 = coord
    #print(np.exp(logP[coord2[:, 0], coord2[:, 1]]))
    coord1 = []
    for c in coord:
        ind = np.where((coord2[:, 0]  == c[0]) * (coord2[:, 1]  == c[1]))
        #if c in coord2:
        if ind[0].size > 0:
            #print(c)
            isIn = intervalOverlap(c, coord2[:]) 
            coord1.append(coord2[isIn][0])
            notIsIn = [not s for s in isIn]
            coord2 = coord2[notIsIn]   
    coord1 = np.array(coord1)  
    if coord1.size == 0:
        plt.show()
        return DirectStretchList

    Probabls = np.exp(logP[coord1[:, 0], coord1[:, 1]])
    # recalibrate probabilites by orthogonal deviation
    if RefineProb:
        PP = []
        for c in coord1:
            xx = x[c[0]:(c[1]+1)]
            yy = y[c[0]:(c[1]+1)]
    #        print(xx.size)
            JJ = [xx[-1] - xx[0], yy[-1] - yy[0]]
            JJ = JJ/np.linalg.norm(JJ)
            JJP = [-JJ[1], JJ[0]]
            rrP= np.array([xx - xx[0], yy - yy[0]]).T@JJP
            rrPmax = np.square(rrP).max()
            rrPth = (xx.size - 1)/8*(ds2.mean()-dr2_error)
            
            PP.append(np.erf(np.sqrt(rrPmax/rrPth/2)))  
        Probabls = np.array(PP)*Probabls
    
        
    coord1 = coord1[Probabls <= maxProbability]
    Probabls = Probabls[Probabls <= maxProbability]
    if coord1.size == 0:
        plt.show()
        return DirectStretchList
    
#    DirectStretchList.append({'Stretch Indices': coord1, 'Probabilities': Probabls, 'Stretch Lengths': np.diff(coord1, axis = 1)[0]})
    # plotting
    if showPlots:
        plt.cla()
        trk.DoPlotXYtrack(marker = 'o')
        for c, P in zip(coord1, Probabls):
            print('P = ' + str(P))
            plt.plot(trk.X_um[c[0]:(c[1]+1)], trk.Y_um[c[0]:(c[1]+1)], marker = 'o', label = 'P = ' + str(P))
        plt.legend()
        plt.show()
    
    for c, P in zip(coord1, Probabls):
        x = trk.X_um[c[0]:(c[1]+1)]
        y = trk.Y_um[c[0]:(c[1]+1)]
        cL, ds = ContourLength(x, y, Rerror = Rerror)
        tStretch = trk.t_s[c[1]] - trk.t_s[c[0]]
        
        DirectStretchList.append({'Stretch Indices': c, 
                                  'Probabilities': P, 
                                  'Stretch Points': np.diff(c)[0],
                                  'X': x,
                                  'Y': y,
                                  'Stretch Length (um)': cL,
                                  'Max Displacement (um)': FindMaxDisplacementOnContour(x, y),
                                  'Stretch Time (s)': tStretch,
                                  'Stretch Speed (um/min)': cL/tStretch*60,
                                  'Displacement Speed (um/min)': FindMaxDisplacementOnContour(x, y)/tStretch*60})
        
    return DirectStretchList, logP

def EstimateDiffusionCoeff(X, Y, t, DirectStretchList, MinStretchLength = 10, Rerror = 0):
    # excludes DirectStretches from the track, and calculates MSD on the rest
    # of the stretches of at least MinStretchLength in length
    dr2_error = Rerror**2
    stretchStart = 0
    MSDlist = []
    MSDerrorList = []
    # sort stretches
    DirectStretchListSorted = sorted(DirectStretchList, key = lambda DirectStretch: DirectStretch['Stretch Indices'][0])

    for DirectStretch in DirectStretchListSorted:
        stretchEnd = DirectStretch['Stretch Indices'][0]
#        print('start = ' + str(stretchStart))
#        print('end = ' + str(stretchEnd))
        if (stretchEnd-stretchStart) >= MinStretchLength:
#            print('here')
            msd = DoCalculateMSD(X = X[stretchStart:stretchEnd], 
                                 Y = Y[stretchStart:stretchEnd], 
                                 t = t[stretchStart:stretchEnd])
            MSDlist.append(msd.XYsq_um2[:MinStretchLength])
            MSDerrorList.append(msd.error_XYsq_um2[:MinStretchLength])
        stretchStart = DirectStretch['Stretch Indices'][1]
    
    stretchEnd = X.size# - 1
    if (stretchEnd-stretchStart) >= MinStretchLength:
        msd = DoCalculateMSD(X = X[stretchStart:stretchEnd], 
                             Y = Y[stretchStart:stretchEnd], 
                             t = t[stretchStart:stretchEnd])
        msd.XYsq_um2[1:] = msd.XYsq_um2[1:] - dr2_error
        MSDlist.append(msd.XYsq_um2[:MinStretchLength])
        MSDerrorList.append(msd.error_XYsq_um2[:MinStretchLength])
        
    MSD, MSDerror = WeightedAverage(MSDlist, MSDerrorList)
    if not np.isnan(MSD).all():
        MSD[0] = 0
        MSDerror[0] = 0
        t = msd.t_s[:MinStretchLength]
    else:
        plt.cla()
        t = np.NaN
        return np.NaN, np.NaN, np.NaN, MSD, MSDerror, t, np.NaN, np.NaN, np.NaN
    
  
    t1 = t[:round(MinStretchLength/2)]
    msd1 = MSD[:round(MinStretchLength/2)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(t1, msd1)
    D1 = slope/4
    print('D linear fit = ' + str(D1))
    plt.cla()
    plt.plot(t, MSD)
    plt.plot(t1, msd1, 'o')
    plt.plot(t1, slope*t1 + intercept, label = 'D = %.2f $\mu$m^2/min' % (D1*60))
    try:
        if slope < 0:
            slope = -slope
        bta, covM =  curve_fit(ConfinedDiffusion2DFit, t1[1:], msd1[1:], p0 = [msd1.max(), slope/msd1.max()] , bounds=([0, 0], [np.inf, np.inf])) #exclude [0,0] point in msd
        print('bta = ' + str(bta))
        confinementRadius = np.sqrt(bta[0])
        D = bta[0]*bta[1]/4
        plt.plot(t1, ConfinedDiffusion2DFit(t1, *bta), label = 
                    'D = %.2f $\mu$m^2/min, Conf radius = %.2f $\mu$m'
                     % (D*60, confinementRadius)) 
    except RuntimeError:
        D = D1
        confinementRadius = np.inf
        print('Confined 2D fit did not succeed. Using the result of linear fitting ')
    
    plt.legend()
    #print('D = ' + str(D))
    #print('ConfinementRadius = ' + str(confinementRadius))
    
    return D, confinementRadius, D1, MSD, MSDerror, t, t1, msd1, intercept

        
def FindConfinements(trk, MinStretchLength = 10, maxProbability = 0.05, Rerror = np.sqrt(0.34**2 + 0.25**2), showPlots = True, Dlims = ()):
    
    dr2_error = Rerror**2
    Jzeros = jn_zeros(0, 2)
    plt.subplot(1, 2, 1)
    D, confinementRadius, D1, MSD, MSDerror, t, t1, msd1, intercept = EstimateDiffusionCoeff(
            trk.X_um, trk.Y_um, trk.t_s, trk.DirectStretchList, MinStretchLength, Rerror = Rerror)
    
    Dlims = np.array(Dlims)/60
    if Dlims.size == 1: # e.g Dlims = (1, )
        D = Dlims[0]
    elif Dlims.size == 3: #e.g. Dlims = (0.2, 1, 4)  in um^2/min !!
        if np.isnan(D):
            D = Dlims[1]
        elif (D < Dlims[0]):
            D = Dlims[0]
        elif (D > Dlims[2]):
            D = Dlims[2]
    print('D = ', D*60)
            
    
    #print('D = ' + str(D))
    ConfinementList = []
    stretchStart = 0
    # sort stretches
    DirectStretchListSorted = sorted(trk.DirectStretchList, key = lambda DirectStretch: DirectStretch['Stretch Indices'][0])
    
    randStretches = []
    for DirectStretch in DirectStretchListSorted:
        stretchEnd = DirectStretch['Stretch Indices'][0]
        randStretches.append([stretchStart, stretchEnd])
        stretchStart = DirectStretch['Stretch Indices'][1]
    stretchEnd = trk.X_um.size
    randStretches.append([stretchStart, stretchEnd])

    for stretch in randStretches:
        stretchStart, stretchEnd = stretch
        if (stretchEnd - stretchStart) >= MinStretchLength:
            print(stretch)
            dist = np.sqrt(GetMaxSqDistMatrix(X = trk.X_um[stretchStart:stretchEnd], 
                                     Y = trk.Y_um[stretchStart:stretchEnd]))
    #        print(distSq.shape)
            binCorrection = 0.57*dist.diagonal(offset = 1).mean() #correction for a small number of steps
            T = trk.t_s[stretchStart:stretchEnd]
            Ta, Tb = np.meshgrid(T, T)
            Tab = np.abs(Ta - Tb)
            I, J = np.meshgrid(np.arange(T.size), np.arange(T.size))
            #logP = 0.2048 - 2.5117*D*Tab/(distSq)# - Rerror**2)
            logP = np.log(2/(Jzeros[0]*besselj1(Jzeros[0]))) - Jzeros[0]**2*D*Tab/((dist + binCorrection)**2-dr2_error)
            logP = logP + np.log(T.size - np.abs(I-J))#np.log((T.size-1)/(np.abs(I-J)-1))
    #        logP[np.isnan(logP)] = logP.max()
            temp = -logP
            
            Nstep = np.arange(T.size)
            Na, Nb = np.meshgrid(Nstep, Nstep)
            Nab = np.abs(Na - Nb)
            cutoffLen = MinStretchLength
            temp[Nab < cutoffLen] = 0
    #        plt.cla()
    #        plt.imshow(temp)
    #        plt.show()
            coord = peak_local_max(temp, threshold_abs = -np.log(3*maxProbability), exclude_border = False)
            if coord.size == 0:
                continue
            ii = np.argsort(-temp[coord[:, 0], coord[:, 1]]) #- sign to have a descending sort
            ii = ii[::2] # remove same peaks from the other matrix triangle
            coord = coord[ii, :]
            coord.sort(axis = 1) #make sure that step indexes in the move are arrange in ascending order
    #        coord = coord[(np.diff(coord, axis =1) >= Llim).T[0]]
            if coord.size == 0:
                plt.show()
                continue     
            # remove overlaping intervals 
            coord2 = coord
            coord1 = []
            for c in coord:
                ind = np.where((coord2[:, 0]  == c[0]) * (coord2[:, 1]  == c[1]))
                #if c in coord2:
                if ind[0].size > 0:
                    #print(c)
                    isIn = intervalOverlap(c, coord2[:]) 
                    coord1.append(coord2[isIn][0])
                    notIsIn = [not s for s in isIn]
                    coord2 = coord2[notIsIn]   
            coord1 = np.array(coord1)  
            if coord1.size == 0:
                plt.show()
                continue
    
            Probabls = np.exp(logP[coord1[:, 0], coord1[:, 1]])       
            coord1 = coord1[Probabls <= maxProbability]
            Probabls = Probabls[Probabls <= maxProbability]
            if coord1.size == 0:
                plt.show()
                continue
            
            for c, P in zip(coord1, Probabls):
                c = c + stretchStart
                x = trk.X_um[c[0]:(c[1]+1)]
                y = trk.Y_um[c[0]:(c[1]+1)]
                cL, ds = ContourLength(x, y, Rerror = Rerror)
                tStretch = trk.t_s[c[1]] - trk.t_s[c[0]]
                
                ConfinementList.append({'Stretch Indices': c, 
                                      'Probabilities': P, 
                                      'Stretch Points': np.diff(c)[0],
                                      'X': x,
                                      'Y': y,
                                      'Stretch Length (um)': cL,
                                      'Max Displacement (um)': FindMaxDisplacementOnContour(x, y),
                                      'Stretch Time (s)': tStretch,
                                      'Stretch Speed (um/min)': cL/tStretch*60,
                                      'Average Diff Coeff (um^2/min)': D*60,
                                      'Confinement from fit (um):': confinementRadius,
                                      'Displacement Speed (um/min)': FindMaxDisplacementOnContour(x, y)/tStretch*60})
    
    
    # plotting
    if showPlots:
        plt.subplot(1, 2, 2)
        plt.cla()
        trk.DoPlotXYtrack(marker = 'o')
        for confinement in ConfinementList:
            P = confinement['Probabilities']
            print('P = ' + str(P))
            x = confinement['X']
            y = confinement['Y']
            plt.plot(x, y, marker = 'o', label = 
                    'P = %.2e, Npoints = %d '
                     % (P, x.size))
        plt.legend()
        plt.show()
    
        
    return ConfinementList, logP
        
    

def FindMaxDisplacementOnContour(x, y, z = 0):
    Xa, Xb = np.meshgrid(x, x)
    Ya, Yb = np.meshgrid(y, y)
    Za, Zb = np.meshgrid(z, z)
    dist = (Xa - Xb)**2 + (Ya - Yb)**2 + (Za - Zb)**2
    return np.sqrt(dist.max())


def ContourLength(x, y, z = np.array(0), Rerror = 0):
    diffX = np.diff(x)
    diffY = np.diff(y)
    if z.size == x.size:
        diffZ = np.diff(z)
    else:
        diffZ = 0
        
    ds2 = diffX**2 + diffY**2 + diffZ**2 - Rerror**2
    ds2[ds2<0] = 0
    ds = np.sqrt(ds2)
    return ds.sum(), ds

def GetMaxSqDistMatrix(X, Y = np.empty((1, 0))):
    # for each two points on the track finds the maximal squared confinement size
    Xa, Xb = np.meshgrid(X, X)
    Xab = (Xa - Xb)
    dist = Xab**2
    if Y.size != 0:
        Ya, Yb = np.meshgrid(Y, Y)
        Yab = (Ya - Yb)
        dist = dist + Yab**2
        
#    for tt in range(2, X.size):
#        d0 = dist.diagonal(offset = tt-1)
#        d1 = dist.diagonal(offset = tt)
#        newdiag = np.array([d0[:-1], d0[1:], d1]).max(axis = 0)
#        dist[range(X.size - tt), range(tt, X.size)] = newdiag
#        dist[range(tt, X.size),  range(X.size - tt)] = newdiag
    
    dist = np.tril(dist)
    for tt in range(X.size - 1):
        dist[tt+1, :] = dist[tt:(tt+2), :].max(axis = 0)
    
    dist = dist + dist.T
    return dist
        

# object saving and loading
def saveObject(obj, *args):
    if len(args) > 0:
        filename = args[0]
    else:
        filename = obj.SaveFileName
        
    if len(filename) < 1:
        print('Provide the filename for saving!')
        return
    
    obj.SaveFileName = filename
    
    with open(filename, 'wb') as ff:
        pickle.dump(obj, ff, pickle.HIGHEST_PROTOCOL)       
        ff.close()
            
        
        

def loadObject(filename):
    with open(filename, 'rb') as ff:
        obj =  pickle.load(ff)
        ff.close()
        return obj


# fit functions
def LinearFit(t, a, b):
    return a*t + b


# Get statistics on np.array          
def GetStats(V, OutlierFrac = 5):
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    lims = np.percentile(V, [OutlierFrac, 100-OutlierFrac])
    V1 = V[(V>=lims[0]) & (V<=lims[1])]
   
    Stats = dict()
    Stats = {'min': V.min(), 'max': V.max(), 'mean': V.mean(), 'std': V.std(),
             'median': np.median(V), 'mad': np.median(np.abs(V - np.median(V))),
             'OutlierFrac': OutlierFrac, 'OutlierLims':lims, 'meanRobust': V1.mean(), 
             'stdRobust': V1.std()}
    return Stats, V1

      
       
        
        