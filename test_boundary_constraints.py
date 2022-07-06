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

from ast import Mult
import numpy as np
import matplotlib.pyplot as plt
import xarray
from floris.simulation.turbine import power
from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane
from floris.tools.floris_interface import generate_heterogeneous_wind_map
import pyoptsparse
import time
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from plotting_functions import plot_poly

def create_eagle_exclusions(filename, threshold):
    r = 6371230.0 # assumed radius of earth

    # lat long of southwest point of flowfield data
    lat0 = 42.944092
    lat = 42.944092
    long = -105.773689
    x_flow = r*np.deg2rad(long) * np.cos(np.deg2rad(lat0))
    y_flow = r*np.deg2rad(lat)

    # lat long of southwest point of eagle data
    lat = 42.78
    long = -106.21
    x_eagle = r*np.deg2rad(long) * np.cos(np.deg2rad(lat0))
    y_eagle = r*np.deg2rad(lat)

    # calculate which indices of eagle data correspond to flowfield
    dx = x_flow - x_eagle
    dy = y_flow - y_eagle
    x_index = int(np.floor(dx/50000 * 1000))
    y_index = int(np.floor(dy/50000 * 1000))

    # load in eagle data
    full_data = np.load(filename)
    excluded_data = full_data[x_index:x_index+120,y_index:y_index+120]
    excluded_data[excluded_data < threshold] = 0
    excluded_data[excluded_data >= threshold] = 1

    nx, ny = np.shape(excluded_data)
    excluded_polygons = MultiPolygon()
    side_x = np.linspace(0.0,6000.0,nx+1)
    side_y = np.linspace(0.0,6000.0,ny+1)
    for i in range(nx):
        for j in range(ny):
            if excluded_data[j,i] == 1:
                added_poly = Polygon(((side_x[i],side_y[j]),(side_x[i+1],side_y[j]),
                                      (side_x[i+1],side_y[j+1]),(side_x[i],side_y[j+1])))
                excluded_polygons = excluded_polygons.union(added_poly)

    if type(excluded_polygons) == Polygon:
        excluded_polygons = MultiPolygon([excluded_polygons])

    return excluded_polygons


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


def boundary_constraint(x, y):

        global excluded_polygons

        point = Point(x, y)
        min_dist = 1E12
        for poly in excluded_polygons:
            distance = poly.exterior.distance(point)
            for interior in poly.interiors:
                interior_distance = interior.distance(point)
                if interior_distance < distance:
                    distance = interior_distance
            if poly.contains(point):
                distance = -distance
            if distance < min_dist:
                min_dist = distance

        return min_dist


if __name__=="__main__":

    global fi
    global start_power
    global scale_x
    global scale_y
    global excluded_polygons

    spatial_array = xarray.open_dataset("inputs/neutral_8mps/wspd_80m_tavg1h_from22200.nc")
    x = spatial_array.x
    y = spatial_array.y
    flow_data = np.loadtxt("inputs/mean_u.txt")

    yy, xx = np.meshgrid(x, y)

    flow_x = np.ndarray.flatten(xx)
    flow_y = np.ndarray.flatten(yy)
    flow_u = [np.ndarray.flatten(flow_data)]

    flow_x = flow_x - np.min(flow_x)
    flow_y = flow_y - np.min(flow_y)

    scale_x = 1.0
    scale_y = 1.0

    het_map_2d = generate_heterogeneous_wind_map(flow_u, flow_x, flow_y)
    fi = FlorisInterface("inputs/birds_siting.yaml", het_map=het_map_2d)


    # xs = np.linspace(np.min(flow_x)+100.0,np.max(flow_x)-100.0,10)
    # ys = np.linspace(np.min(flow_y)+100.0,np.max(flow_y)-100.0,10)
    # start_x, start_y = np.meshgrid(xs, ys)
    # start_x = np.ndarray.flatten(start_x)
    # start_y = np.ndarray.flatten(start_y)

    # fi.reinitialize(layout=[start_x,start_y])
    # fi.calculate_wake()
    # start_power = np.sum(fi.get_farm_power())

    # optProb = pyoptsparse.Optimization("optimize baseline", objective_function)

    # optProb.addVarGroup("turbine_x", len(start_x), type="c", value=start_x/scale_x, upper=np.max(start_x)/scale_x, lower=np.min(start_x)/scale_x)
    # optProb.addVarGroup("turbine_y", len(start_y), type="c", value=start_y/scale_y, upper=np.max(start_y)/scale_y, lower=np.min(start_y)/scale_y)

    # rotor_diameter = 77.0
    # min_spacing = 4.0
    # optProb.addCon("spacing_con",lower=min_spacing*rotor_diameter)

    # optProb.addObj("obj")

    # optimize = pyoptsparse.SNOPT()

    # optimize.setOption("MAXIT",value=5)
    # optimize.setOption("ACC",value=1E-5)
    # optimize.setOption("Major iterations limit", value=5)

    # start_opt = time.time()
    # solution = optimize(optProb,sens="FD", storeHistory="run10x10.hst")
    # optimization_time = time.time()-start_opt
    # print("optimization time: ", time.time()-start_opt)
    # print("end optimization")

    # # END RESULTS
    # opt_DVs = solution.getDVs()
    # opt_x = opt_DVs["turbine_x"] * scale_x
    # opt_y = opt_DVs["turbine_y"] * scale_y

    # print("turbine_x = np." + "%s"%repr(opt_x))
    # print("turbine_y = np." + "%s"%repr(opt_y))

    opt_x = [0.0]
    opt_y = [0.0]
    fi.reinitialize(layout=[opt_x,opt_y])
    fi.calculate_wake()
    farm_power = fi.get_farm_power()

    # print("optimized power: ", np.sum(farm_power))
    # print("optimized objective: ", -np.sum(farm_power)/start_power)


    # Using the FlorisInterface functions for generating plots, run FLORIS
    # and extract 2D planes of data.
    horizontal_plane_2d = fi.calculate_horizontal_plane(80.0, x_bounds=(np.min(flow_x),np.max(flow_x)),y_bounds=(np.min(flow_y),np.max(flow_y)),x_resolution=200, y_resolution=100)

    # Create the plots

    # plot resource
    # ax = plt.gca()
    # cm = ax.pcolormesh(xx,yy,flow_data,shading='auto')

    # plot flow field 
    # plt.figure(1)
    # ax1 = plt.gca()
    # visualize_cut_plane(horizontal_plane_2d, ax=ax1, title="farm flow field", color_bar=True)
    # ax1.set_xlabel('x'); ax1.set_ylabel('y')

    # plt.show()

    # flow field
    lat0 = 42.944092
    lat = 42.944092
    long = -105.773689

    r = 6371230.0
    x_flow = r*np.deg2rad(long) * np.cos(np.deg2rad(lat0))
    y_flow = r*np.deg2rad(lat)

    # eagle data
    # (-106.21, 42.78)
    lat = 42.78
    long = -106.21
    x_eagle = r*np.deg2rad(long) * np.cos(np.deg2rad(lat0))
    y_eagle = r*np.deg2rad(lat)


    dx = x_flow - x_eagle
    dy = y_flow - y_eagle
    x_index = int(np.floor(dx/50000 * 1000))
    y_index = int(np.floor(dy/50000 * 1000))


    full_data = np.load("eagle_probability_data.npy")
    excluded_polygons = full_data[x_index:x_index+120,y_index:y_index+120]
    
    plt.figure(1)
    plt.imshow(excluded_polygons, origin="lower")
    # plt.title("excluded data")

    plt.figure(2)
    plt.imshow(full_data, origin="lower")
    # plt.title("excluded data")

    # plt.figure(3)
    # thresh = 0.2
    # a = np.copy(excluded_polygons)
    # a[a < thresh] = 0.0
    # a[a >= thresh] = 1.0
    # plt.imshow(a, origin="lower")
    # plt.title("thresh = %s"%thresh)

    # excluded_polygons = create_eagle_exclusions("eagle_probability_data.npy", thresh)
    # print("boundary constraint: ", boundary_constraint(100.0,100.0))
    # print("boundary constraint: ", boundary_constraint(475.0,1350.0))
    # print("boundary constraint: ", boundary_constraint(2500.0,3200.0))
    # print("boundary constraint: ", boundary_constraint(2000.0,5600.0))
    # plot_poly(excluded_polygons, ax=ax1)
    plt.show()