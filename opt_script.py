# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import matplotlib.pyplot as plt
import xarray
from floris.simulation.turbine import power
from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane
from floris.tools.floris_interface import generate_heterogeneous_wind_map
import pyoptsparse
import time

"""
This example initializes the FLORIS software, and then uses internal
functions to run a simulation and plot the results. In this case,
we are plotting three slices of the resulting flow field:
1. Horizontal slice parallel to the ground and located at the hub height
2. Vertical slice of parallel with the direction of the wind
3. Veritical slice parallel to to the turbine disc plane
"""


def calc_spacing(layout_x,layout_y):

        nTurbs = len(layout_x)
        npairs = int((nTurbs*(nTurbs-1))/2)
        spacing = np.zeros(npairs)

        ind = 0
        for i in range(nTurbs):
            for j in range(i,nTurbs):
                if i != j:
                    spacing[ind] = np.sqrt((layout_x[i]-layout_x[j])**2+(layout_y[i]-layout_y[j])**2)
                    ind += 1

        return spacing


def objective_function(x):

    global fi
    global start_power
    global scale_x
    global scale_y

    turbine_x = x["turbine_x"]*scale_x
    turbine_y = x["turbine_y"]*scale_y

    fi.reinitialize(layout=[turbine_x,turbine_y])
    fi.calculate_wake()
    farm_power = fi.get_farm_power()
    spacing = calc_spacing(turbine_x, turbine_y)
    fail = False
    funcs = {}
    funcs["obj"] = -np.sum(farm_power)/start_power
    funcs["spacing_con"] = np.min(spacing)

    return funcs, fail


if __name__=="__main__":

    global fi
    global start_power
    global scale_x
    global scale_y

    spatial_array = xarray.open_dataset("inputs/neutral_8mps/wspd_80m_tavg1h_from22200.nc")
    x = spatial_array.x
    y = spatial_array.y
    flow_data = np.loadtxt("inputs/mean_u.txt")

    yy, xx = np.meshgrid(x, y)

    flow_x = np.ndarray.flatten(xx)
    flow_y = np.ndarray.flatten(yy)
    flow_u = [np.ndarray.flatten(flow_data)]

    # scale_x = np.max(flow_x)
    # scale_y = -np.min(flow_y)
    scale_x = 1.0
    scale_y = 1.0

    het_map_2d = generate_heterogeneous_wind_map(flow_u, flow_x, flow_y)
    fi = FlorisInterface("inputs/birds_siting.yaml", het_map=het_map_2d)

    xs = np.linspace(np.min(flow_x)+100.0,np.max(flow_x)-100.0,10)
    print((xs[1]-xs[0])/77.0)
    ys = np.linspace(np.min(flow_y)+100.0,np.max(flow_y)-100.0,10)
    start_x, start_y = np.meshgrid(xs, ys)
    start_x = np.ndarray.flatten(start_x)
    start_y = np.ndarray.flatten(start_y)

    fi.reinitialize(layout=[start_x,start_y])
    fi.calculate_wake()
    start_power = np.sum(fi.get_farm_power())

    optProb = pyoptsparse.Optimization("optimize baseline", objective_function)

    optProb.addVarGroup("turbine_x", len(start_x), type="c", value=start_x/scale_x, upper=np.max(start_x)/scale_x, lower=np.min(start_x)/scale_x)
    optProb.addVarGroup("turbine_y", len(start_y), type="c", value=start_y/scale_y, upper=np.max(start_y)/scale_y, lower=np.min(start_y)/scale_y)

    rotor_diameter = 77.0
    min_spacing = 4.0
    optProb.addCon("spacing_con",lower=min_spacing*rotor_diameter)

    optProb.addObj("obj")

    optimize = pyoptsparse.SNOPT()
    # optimize.setOption("MAXIT",value=5)
    # optimize.setOption("ACC",value=1E-5)
    # optimize.setOption("Major iterations limit", value=5)

    start_opt = time.time()
    solution = optimize(optProb,sens="FD", storeHistory="run10x10.hst")
    optimization_time = time.time()-start_opt
    print("optimization time: ", time.time()-start_opt)
    print("end optimization")

    # END RESULTS
    opt_DVs = solution.getDVs()
    opt_x = opt_DVs["turbine_x"] * scale_x
    opt_y = opt_DVs["turbine_y"] * scale_y

    print("turbine_x = np." + "%s"%repr(opt_x))
    print("turbine_y = np." + "%s"%repr(opt_y))

    fi.reinitialize(layout=[opt_x,opt_y])
    fi.calculate_wake()
    farm_power = fi.get_farm_power()

    print("optimized power: ", np.sum(farm_power))
    print("optimized objective: ", -np.sum(farm_power)/start_power)


    # Using the FlorisInterface functions for generating plots, run FLORIS
    # and extract 2D planes of data.
    horizontal_plane_2d = fi.calculate_horizontal_plane(x_bounds=(np.min(flow_x),np.max(flow_x)),y_bounds=(np.min(flow_y),np.max(flow_y)),x_resolution=200, y_resolution=100)

    # Create the plots

    # plot resource
    # ax = plt.gca()
    # cm = ax.pcolormesh(xx,yy,flow_data,shading='auto')

    # plot flow field 
    ax = plt.gca()
    visualize_cut_plane(horizontal_plane_2d, ax=ax, title="farm flow field", color_bar=True)
    ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.show()
