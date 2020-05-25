#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:20:00 2018

@author: Oleg Krichevsky okrichev@bgu.ac.il
"""

import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'toolmanager'
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from scipy.optimize import curve_fit
import numpy as np
import os
#from CellTrackModule import CellTrackClass


class FitTool(ToolBase):
    
    
    def __init__(self, toolmanager, name, line, fitName, dataObj, **kwargs):
        super(FitTool, self ).__init__(toolmanager, name)
        self.line = line
        self.FitParam = dict()
#        self.dataErrs = dataErrs
        self.fitName = fitName
        self.dataObj = dataObj
        self.params = kwargs
#        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    # keyboard shortcut
#    default_keymap = 'f'
    description = 'Fit Tool'
    image = '/Users/oleg/Documents/Python programming/NYU/FitToolIcon.png'


    def trigger(self, sender, event, data=None):
#        xs = self.line.get_xdata()
#        ys = self.line.get_ydata()

        XLim = self.line.axes.get_xlim()
        YLim = self.line.axes.get_ylim()
        self.dataObj.DoFitMSDLims(fitName = self.fitName, XLim = XLim , YLim = YLim, **self.params)
        self.line.remove()
        plt.gcf().canvas.draw()
        plt.show()


def curvefitLims(fitName, xs, ys, ysErrs, XLim, YLim):
    inLims = (xs >= XLim[0]) & (xs <= XLim[1]) & (ys >= YLim[0]) & (ys <= YLim[1]) 
    isfiniteErr =  (ysErrs > 0) & np.isfinite(ysErrs)
    #print(inLims)
    x = xs[inLims & isfiniteErr]
#    print(x)
    y = ys[inLims & isfiniteErr]
    yErr = ysErrs[inLims & isfiniteErr]
#    print(yErr)
    fitFunc = globals()[fitName]
    print(fitName)
    bta, covM = curve_fit(fitFunc, x, y, sigma=yErr, absolute_sigma = True)
#    print(bta)
    FitParam = dict()
    FitParam['beta'] = bta
    FitParam['errorBeta'] = np.sqrt(np.diag(covM))
    chiSqArray = np.square((fitFunc(x, *bta) - y)/yErr)
    FitParam['chiSqNorm'] =  chiSqArray.sum()/x.size
    FitParam['x'] = x
    FitParam['y'] = y
    FitParam['XLim'] = XLim
    FitParam['YLim'] = YLim
    #print(FitParam)
    plt.errorbar(xs, ys, ysErrs, fmt = '.')
    plt.plot(xs[inLims], fitFunc(xs[inLims], *bta)) 
    plt.gcf().canvas.draw()
    plt.autoscale()
    plt.show()

    
    return FitParam


# Fit functions      
def LinearFit(t, a, b):
    return a*t + b        
        
def PowerFit(t, a, n):
    return a*t**n     

def BallisticDiffusion2DFit(t, D, tau):
    MSD = 4*D*(t - tau*(1 - np.exp(- t / tau)))
    return MSD

def ConfinedDiffusion2DFit(t, Rsq, k):
    # Rsq is the effective radius of the confinement
    # k = 4*D/Rsq
    # MSDerror
    MSD = Rsq*(1 - np.exp(- k*t))
    return MSD


class NextItemTool(ToolBase):
       
    def __init__(self, toolmanager, name, dataListObj, methodName, **kwargs):
        super(NextItemTool, self ).__init__(toolmanager, name)
 #       ToolBase.__init__(toolmanager, name)
        self.currentItem = 0
        self.dataListObj = dataListObj
        self.methodName = methodName
        self.params = kwargs #parameters for the method
        
        whatToDo = getattr(self.dataListObj[self.currentItem], self.methodName)
        whatToDo(**self.params)


    description = 'Next Item Tool'
    image = '/Users/oleg/Documents/Python programming/NYU/Forward arrow.png'


    def trigger(self, sender, event, data=None):
        if self.currentItem < (len(self.dataListObj)-1):
            plt.cla()
            self.currentItem = self.currentItem + 1
            whatToDo = getattr(self.dataListObj[self.currentItem], self.methodName)
            whatToDo(**self.params)
            self.toolmanager.tools['Previous Item'].currentItem = self.currentItem
            plt.title('Item No = ' + str(self.currentItem) + ' of ' + str(len(self.dataListObj)))
        else:
            plt.title('End of the list reached')
            os.system( "say end reached" )
        plt.gcf().canvas.draw()
        plt.show()
        

class PreviousItemTool(ToolBase):
       
    def __init__(self, toolmanager, name, dataListObj, methodName, **kwargs):
        super(PreviousItemTool, self ).__init__(toolmanager, name)
#        ToolBase.__init__(toolmanager, name)
        self.currentItem = 0
        self.dataListObj = dataListObj
        self.methodName = methodName
        self.params = kwargs #parameters for the method
        
#        whatToDo = getattr(self.dataListObj[self.currentItem], self.methodName)
#        whatToDo(**self.params)


    description = 'Next Item Tool'
    image = '/Users/oleg/Documents/Python programming/NYU/Backward arrow.png'


    def trigger(self, sender, event, data=None):
        if self.currentItem > 0:
            self.currentItem = self.currentItem - 1
            plt.cla()
            whatToDo = getattr(self.dataListObj[self.currentItem], self.methodName)
            whatToDo(**self.params)
            self.toolmanager.tools['Next Item'].currentItem = self.currentItem
            plt.title('Item No = ' + str(self.currentItem) + ' of ' + str(len(self.dataListObj)))
        else:
            plt.title('Start of the list reached')
            os.system( "say start reached" )
        plt.gcf().canvas.draw()
        plt.show()
        
def WeightedAverage(ListOfDataVectors, ListOfDataErrors):
    DataVectors = np.array(ListOfDataVectors)
    DataErrors = np.array(ListOfDataErrors)
    weights = DataErrors**(-2)
    WA = np.sum(DataVectors*weights, 0)/np.sum(weights, 0)
    WE = np.sum(weights, 0)**(-2)
    return WA, WE