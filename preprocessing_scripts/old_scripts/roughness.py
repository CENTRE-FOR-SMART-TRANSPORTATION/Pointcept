#!/usr/bin/env python3

##########################################################################
#                                                                        #
#                              CloudComPy                                #
#                                                                        #
#  This program is free software; you can redistribute it and/or modify  #
#  it under the terms of the GNU General Public License as published by  #
#  the Free Software Foundation; either version 3 of the License, or     #
#  any later version.                                                    #
#                                                                        #
#  This program is distributed in the hope that it will be useful,       #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
#  GNU General Public License for more details.                          #
#                                                                        #
#  You should have received a copy of the GNU General Public License     #
#  along with this program. If not, see <https://www.gnu.org/licenses/>. #
#                                                                        #
#          Copyright 2020-2021 Paul RASCLE www.openfields.fr             #
#                                                                        #
##########################################################################

import os
import sys
import math
import numpy as np
import time

from gendata import getSampleCloud2, getSamplePoly2, dataDir, isCoordEqual, createSymbolicLinks
import cloudComPy as cc

createSymbolicLinks() # required for tests on build, before cc.initCC

cloud = cc.loadPointCloud(os.path.join(os.path.expanduser("~"), "Desktop", "las_files", "03702E_C1R1_R1R1_18000_20000_section_12_sar_d_points_35.las"))
print(cloud.isShifted())
print(cloud.getGlobalShift())
print(cloud)
nsf = cloud.getNumberOfScalarFields()
print(nsf)
coordinates = cloud.toNpArrayCopy()    
print(coordinates)
x=coordinates[:,0]                                                     # x column
y=coordinates[:,1]
z=coordinates[:,2]
print(x)
print(len(x))
d = cloud.getScalarFieldDic()
print(d)
timestamps=[]
timestamps.append(time.time())

ret = cc.computeRoughness(0.06, [cloud])
if not ret:
    raise RuntimeError
timestamps.append(time.time())
print("duration computeRoughness:", timestamps[-1] -timestamps[-2])
print("Total duration:", timestamps[-1] -timestamps[0])

ret = cc.computeRoughness(0.06, [cloud], (0., 1., 0.))
if not ret:
    raise RuntimeError
timestamps.append(time.time())
print("duration computeRoughness up dir:", timestamps[-1] -timestamps[-2])
print("Total duration:", timestamps[-1] -timestamps[0])

print(ret)

nsf = cloud.getNumberOfScalarFields()
print(nsf)
d = cloud.getScalarFieldDic()
print(d)

roughness = cloud.getScalarField(nsf-1)
r = roughness.toNpArrayCopy()
print(r, len(r))

with open("hello.txt", "w") as f:
    for a, b, c, d in zip(x, y, z, r):
        f.write(f"{a},{b},{c},{d}\n")


ret = cc.computeLocalDensity(cc.Density.DENSITY_KNN, 0.06, [cloud])
if not ret:
    raise RuntimeError
timestamps.append(time.time())

nsf = cloud.getNumberOfScalarFields()
print(nsf)
d = cloud.getScalarFieldDic()
print(d)

roughness = cloud.getScalarField(nsf-1)
r = roughness.toNpArrayCopy()
print(r, len(r))


ok = cloud.exportCoordToSF(False, False, True) # Z coordinate as a scalar Field


nsf = cloud.getNumberOfScalarFields()
print(nsf)

cloud.computeScalarFieldGradient(nsf-1, 0.06, True)

nsf = cloud.getNumberOfScalarFields()
print(nsf)

d = cloud.getScalarFieldDic()
print(d)

roughness = cloud.getScalarField(nsf-1)
r = roughness.toNpArrayCopy()
print(r, len(r))

cloud.renameScalarField(8, "Density")

d = cloud.getScalarFieldDic()
print(d)

ret = cc.SavePointCloud(cloud, "cloud.las")
