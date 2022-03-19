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

    outer_poly = Polygon(((-5000,-5000),(11000,-5000),(11000,11000),(-5000,11000)))
    inner_poly = Polygon(((0,0),(6000,0),(6000,6000),(0,6000)))
    boundary_poly = outer_poly.difference(inner_poly)
    excluded_polygons = excluded_polygons.union(boundary_poly)
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

    nturbs = len(turbine_x)
    boundary_constraint = np.zeros(nturbs)
    for i in range(nturbs):
        boundary_constraint[i] = calc_boundary_constraint(turbine_x[i], turbine_y[i])

    fi.reinitialize(layout=[turbine_x,turbine_y])
    fi.calculate_wake()
    farm_power = fi.get_farm_power()
    spacing = calc_spacing(turbine_x, turbine_y)
    fail = False
    funcs = {}
    funcs["obj"] = -np.sum(farm_power)/start_power
    funcs["spacing_con"] = np.min(spacing)
    funcs["boundary_con"] = boundary_constraint

    return funcs, fail


def calc_boundary_constraint(x, y):

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

    minx = np.min(flow_x)+100.0
    maxx = np.max(flow_x)-100.0
    miny = np.min(flow_y)+100.0
    maxy = np.max(flow_y)-100.0

    start_x = np.array([-400.        , 1340.74743765, 1640.00619292, 2570.77817285,
       3030.2188474 , 5480.        , 5480.        , -400.        ,
        706.94273508, 1769.98067136, 2086.85995351, 3789.45326298,
       4589.9999132 , 5480.        , -400.        ,  570.00805934,
       1740.74169182, 2259.99907449, 3260.00003451, 4580.00142738,
       5480.        , -400.        ,   60.00462569, 1809.99457246,
       2699.99769248, 3535.14306046, 4490.05819115, 5480.        ,
       -400.        ,  539.99948882, 1560.00025661, 3060.00005273,
       3410.00121144, 3670.00888214, 5480.        , -399.9998083 ,
        580.00052014, 2066.80086422, 2359.93937375, 3610.00299294,
       4360.00059864, 5480.        , -400.        , 1460.00023172,
       1918.39138441, 2208.93642925, 3000.00662206, 3969.99703028,
       5480.        ])
    start_y = np.array([-3400.        , -3400.        , -3150.76464536, -3400.        ,
       -3269.996863  , -3330.00564251, -2986.62229316, -2639.25090319,
       -1901.7978694 , -2788.83912134, -2439.14332026, -2054.06047365,
       -2299.24766487, -2678.62177524, -1458.61874277, -1339.25654382,
       -1789.99264804, -1680.75424662,  -819.25031856, -1460.73526356,
       -1309.23671135,  -113.08877267,  -220.74756859,  -359.24983197,
         109.24689883,  -680.82941535,  -558.27775795,  -259.26837141,
         200.74984294,   450.74948686,   570.75011145,   469.04069253,
         697.32305545,   350.75133144,   949.21580204,  1209.23883688,
        1409.25497851,  1593.26112492,  1687.78014592,  1170.75002127,
        1819.99972556,  1419.21485413,  2480.        ,  2369.21652038,
        2076.85836972,  1974.63064575,  2216.52247179,  2480.        ,
        2480.        ])
    
    start_x = start_x - np.min(start_x)
    start_y = start_y - np.min(start_y)

    fi.reinitialize(layout=[start_x,start_y])
    fi.calculate_wake()
    start_power = np.sum(fi.get_farm_power())

    optProb = pyoptsparse.Optimization("optimize baseline", objective_function)

    optProb.addVarGroup("turbine_x", len(start_x), type="c", value=start_x/scale_x, upper=maxx/scale_x, lower=minx/scale_x)
    optProb.addVarGroup("turbine_y", len(start_y), type="c", value=start_y/scale_y, upper=maxy/scale_y, lower=miny/scale_y)

    rotor_diameter = 77.0
    min_spacing = 4.0
    thresh = 0.5
    excluded_polygons = create_eagle_exclusions("eagle_probability_data.npy", thresh)
    # full_data = np.load("eagle_probability_data.npy")
    # x_index = 710
    # y_index = 364
    # excluded_data = full_data[x_index:x_index+120,y_index:y_index+120]
    # x = np.ndarray.flatten(excluded_data)
    # print("50: ", np.percentile(x,50))
    # print("75: ", np.percentile(x,25))
    # print("80: ", np.percentile(x,20))
    # print("90: ", np.percentile(x,15))
    # print("95: ", np.percentile(x,10))
    optProb.addCon("spacing_con",lower=min_spacing*rotor_diameter)
    optProb.addConGroup("boundary_con",len(start_x),lower=0.0)

    optProb.addObj("obj")

    optimize = pyoptsparse.SNOPT()
    # optimize.setOption("MAXIT",value=5)
    # optimize.setOption("ACC",value=1E-5)
    # optimize.setOption("Major iterations limit", value=5)

    start_opt = time.time()
    solution = optimize(optProb,sens="FD", storeHistory="run7x7_0.5.hst")
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


    # # Using the FlorisInterface functions for generating plots, run FLORIS
    # # and extract 2D planes of data.
    horizontal_plane_2d = fi.calculate_horizontal_plane(80.0, x_bounds=(np.min(flow_x),np.max(flow_x)),y_bounds=(np.min(flow_y),np.max(flow_y)),x_resolution=200, y_resolution=100)

    # # Create the plots

    # # plot resource
    # ax = plt.gca()
    # cm = ax.pcolormesh(xx,yy,flow_data,shading='auto')

    # # plot flow field 
    plt.figure(1)
    ax1 = plt.gca()
    visualize_cut_plane(horizontal_plane_2d, ax=ax1, title="farm flow field", color_bar=True)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    plot_poly(excluded_polygons, ax=ax1)
    plt.show()