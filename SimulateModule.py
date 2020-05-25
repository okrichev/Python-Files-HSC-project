#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:15:31 2019

@author: Oleg Krichevsky okrichev@bgu.ac.il
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
from CellTrackModule import CellTrackClass
from CellTrackModule import CellTrackArrayClass


def CreateRandomWalk(Nsteps, AveStepSize):
    dX = AveStepSize*np.random.normal(size = Nsteps)
    X = np.cumsum(dX)
    dY = AveStepSize*np.random.normal(size = Nsteps)
    Y = np.cumsum(dY)
    dZ = AveStepSize*np.random.normal(size = Nsteps)
    Z = np.cumsum(dZ)
    return X, Y, Z

def CreateRadiallyConfined2DRandomWalk(Nsteps, AveStepSize, Radius, StartLocation = [0, 0]):
    # square/cubic confinement
    dX = AveStepSize*np.random.normal(size = Nsteps)
    X = np.cumsum(np.insert(dX, 0, 0)) + StartLocation[0]
    dY = AveStepSize*np.random.normal(size = Nsteps)
    Y = np.cumsum(np.insert(dY, 0, 0)) + StartLocation[1]
    Xi = np.copy(X)
    Yi = np.copy(Y)
#    dZ = AveStepSize*np.random.normal(size = Nsteps)
#    Z = np.cumsum(dZ)
    Rsq = X**2 + Y**2
    indRR = np.where(Rsq > Radius**2)
    #print(indRR)
    while indRR[0].size > 0:
        indR = indRR[0][0]
#    print(Rsq)
        #print('indR = '+ str(indR))
    
        HitSegm = np.array([(indR-1), indR])
        Xc, Yc, RotCCMtrx, RotCWMtrx = FindSegmentCircleCrossing(X[HitSegm], Y[HitSegm], Radius)
        Xr, Yr = ReflectTrajectory(X[indR:], Y[indR:], Xc, Yc, RotCCMtrx, RotCWMtrx)
        X[indR:] = Xr 
        Y[indR:] = Yr
        
        Rsq = X**2 + Y**2
        indRR = np.where(Rsq > Radius**2)

    return X, Y, Xi, Yi

def FindSegmentCircleCrossing(X, Y, Radius):
    dx = np.diff(X)
    dy = np.diff(Y)
    dr2 = dx**2 + dy**2
    D = X[0]*Y[1] - X[1]*Y[0]
    Xc0 = (D*dy + np.sign(dy)*dx*np.sqrt(Radius**2*dr2 - D**2))/dr2
    Xc1 = (D*dy - np.sign(dy)*dx*np.sqrt(Radius**2*dr2 - D**2))/dr2
    Yc0 = (-D*dx + np.abs(dy)*np.sqrt(Radius**2*dr2 - D**2))/dr2
    Yc1 = (-D*dx - np.abs(dy)*np.sqrt(Radius**2*dr2 - D**2))/dr2
    if dy > 0:
        Xc, Yc = Xc0[0], Yc0[0]
    else:
        Xc, Yc = Xc1[0], Yc1[0]
    
    RotCCMtrx = np.array([[Xc, -Yc], [Yc, Xc]])/np.sqrt(Xc**2 + Yc**2)
    RotCWMtrx = np.array([[Xc, Yc], [-Yc, Xc]])/np.sqrt(Xc**2 + Yc**2)
    #print('shape = ' + str(np.array([[Xc, -Yc], [Yc, Xc]]).shape))
    
    return Xc, Yc, RotCCMtrx, RotCWMtrx

def ReflectTrajectory(X, Y, Xc, Yc, RotCCMtrx, RotCWMtrx):
    # rotate clockwise  
    #print(np.array([X - Xc, Y - Yc]))
#    print([X - Xc, Y - Yc])
    XYrotated  = RotCWMtrx @ np.array([X - Xc, Y - Yc])
#    print(XYrotated)
    # reflect from Y
    XYreflected = np.array([-XYrotated[0, :], XYrotated[1, :]])
#    print(XYreflected)
    # rotate and translate back
    XYfin = RotCCMtrx @ XYreflected
    Xfin = XYfin[0, :] + Xc
    Yfin = XYfin[1, :] + Yc
    
    return Xfin, Yfin
    

def CreateRConfined2D_RW_CellTrackArrayClass(Ntracks = 10, Nsteps = 10, AveStepSize = 1, 
                                             Radius = 3, StartLocation = [0, 0]):
    SimTrksConfined = CellTrackArrayClass()
    SimTrksRW = CellTrackArrayClass()
#    trk = CellTrackClass()
#    trk.t_s = np.arange(0, Nsteps)
#    trk.Z_um = np.zeros(Nsteps)
    
    for i in range(Ntracks):
        print('trackNo = ' + str(i))
        trk = CellTrackClass()
        trk.t_s = np.arange(0, Nsteps+1)
        trk.Z_um = np.zeros(Nsteps+1)        
        X, Y, Xi, Yi = CreateRadiallyConfined2DRandomWalk(Nsteps, AveStepSize, Radius, StartLocation)
        trk.X_um = X
        trk.Y_um = Y
        SimTrksConfined.tracks.append(trk)

        trk = CellTrackClass()
        trk.t_s = np.arange(0, Nsteps+1)
        trk.Z_um = np.zeros(Nsteps+1)
        trk.X_um = Xi
        trk.Y_um = Yi
        SimTrksRW.tracks.append(trk)
     
    SimTrksConfined.DoTrackCalculations()    
    SimTrksRW.DoTrackCalculations() 
    return SimTrksConfined, SimTrksRW  