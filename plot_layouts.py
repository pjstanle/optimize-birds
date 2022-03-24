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
import time
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from plotting_functions import plot_poly, plot_turbines

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

    het_map_2d = generate_heterogeneous_wind_map(flow_u, flow_x, flow_y)
    fi = FlorisInterface("inputs/birds_siting.yaml", het_map=het_map_2d)

    rotor_diameter = 77.0
    thresh = 0.3
    excluded_polygons = create_eagle_exclusions("eagle_probability_data.npy", thresh)

    # BASELINE
    # p = 27881025.235700626
    # turbine_x = np.array([-400.        , 1340.74743765, 1640.00619292, 2570.77817285,
    #    3030.2188474 , 5480.        , 5480.        , -400.        ,
    #     706.94273508, 1769.98067136, 2086.85995351, 3789.45326298,
    #    4589.9999132 , 5480.        , -400.        ,  570.00805934,
    #    1740.74169182, 2259.99907449, 3260.00003451, 4580.00142738,
    #    5480.        , -400.        ,   60.00462569, 1809.99457246,
    #    2699.99769248, 3535.14306046, 4490.05819115, 5480.        ,
    #    -400.        ,  539.99948882, 1560.00025661, 3060.00005273,
    #    3410.00121144, 3670.00888214, 5480.        , -399.9998083 ,
    #     580.00052014, 2066.80086422, 2359.93937375, 3610.00299294,
    #    4360.00059864, 5480.        , -400.        , 1460.00023172,
    #    1918.39138441, 2208.93642925, 3000.00662206, 3969.99703028,
    #    5480.        ])
    # turbine_y = np.array([-3400.        , -3400.        , -3150.76464536, -3400.        ,
    #    -3269.996863  , -3330.00564251, -2986.62229316, -2639.25090319,
    #    -1901.7978694 , -2788.83912134, -2439.14332026, -2054.06047365,
    #    -2299.24766487, -2678.62177524, -1458.61874277, -1339.25654382,
    #    -1789.99264804, -1680.75424662,  -819.25031856, -1460.73526356,
    #    -1309.23671135,  -113.08877267,  -220.74756859,  -359.24983197,
    #      109.24689883,  -680.82941535,  -558.27775795,  -259.26837141,
    #      200.74984294,   450.74948686,   570.75011145,   469.04069253,
    #      697.32305545,   350.75133144,   949.21580204,  1209.23883688,
    #     1409.25497851,  1593.26112492,  1687.78014592,  1170.75002127,
    #     1819.99972556,  1419.21485413,  2480.        ,  2369.21652038,
    #     2076.85836972,  1974.63064575,  2216.52247179,  2480.        ,
    #     2480.        ])

    # 0.5
    # p = 28089774.78534683
    # turbine_x = np.array([ 100.        , 1743.54326602, 1972.87240875, 2990.82480219,
    #    3509.53679906, 5978.68914532, 5980.        ,  100.        ,
    #    1230.98586538, 2212.29951211, 2528.36855992, 4233.96642164,
    #    4997.74735894, 5980.        ,  100.        ,  989.15309633,
    #    2233.41979363, 2607.60678994, 3621.57449821, 4995.92126503,
    #    5980.        ,  100.        ,  562.52297236, 2302.27130316,
    #    3149.03964443, 4029.81855086, 4810.3606104 , 5980.        ,
    #     100.        ,  899.02584417, 1999.80750343, 3235.92642767,
    #    3750.38358719, 4021.97710474, 5980.        ,  105.0499532 ,
    #     852.65307171, 2573.50171149, 2842.06666895, 3987.7033155 ,
    #    4727.5413497 , 5899.28445783,  100.        , 1853.62817261,
    #    2346.30676881, 2594.90189866, 3297.69413472, 4413.35015594,
    #    5916.06322347])
    # turbine_y = np.array([ 100.        ,  100.02017055,  439.20117128,  211.23858491,
    #     313.84820265,  100.8092493 ,  485.84676238,  771.71024458,
    #    1462.6868173 ,  623.9841052 ,  977.52224554, 1308.71758352,
    #    1177.95807831,  784.37259309, 1924.3784177 , 2077.91649709,
    #    1599.47213253, 1709.38858352, 2633.84737497, 1964.9586627 ,
    #    2090.74985112, 3347.04570662, 3253.39347579, 3135.3662711 ,
    #    3481.44881047, 2793.60345531, 2910.54292814, 3239.22687845,
    #    3645.93444657, 3782.66806152, 3995.60733029, 3890.0295553 ,
    #    4128.16545323, 3689.09945309, 4423.87176439, 4522.99022163,
    #    4675.75625325, 5014.68698123, 5144.96482298, 4591.06207087,
    #    5230.04586623, 4734.01860814, 5980.        , 5717.18086092,
    #    5477.73033597, 5312.41418779, 5599.01686911, 5980.        ,
    #    5848.21229158])

    # 0.4
    # p = 27860288.58603824
    # turbine_x = np.array([ 100.00421827, 1669.22703853, 1983.9463635 , 2940.98203755,
    #    3449.75379907, 5919.18884429, 5979.89719195,  100.        ,
    #    1190.87834212, 2164.19520957, 2636.62577704, 4223.52866021,
    #    4892.64759313, 5980.        ,  100.        , 1006.35369784,
    #    2163.53423314, 2531.20211846, 3621.84387995, 4999.84722762,
    #    5970.83213049,  100.        ,  545.69557391, 2243.53801074,
    #    3143.48639815, 4015.34001047, 4877.0551868 , 5970.66618292,
    #     100.        ,  912.25425881, 1985.57915205, 3242.19314619,
    #    3749.89795747, 4029.47973584, 5934.83172986,  100.        ,
    #     840.7294424 , 2532.91410553, 2808.42001409, 3985.31763001,
    #    4727.10263708, 5890.95456887,  100.        , 1871.82859686,
    #    2301.17052358, 2582.0023095 , 3345.20980919, 4375.81287211,
    #    5910.27421722])
    # turbine_y = np.array([ 100.        ,  246.25103413,  332.13612459,  193.34593531,
    #     100.        ,  104.66081941,  403.69551274,  766.10064293,
    #    1479.22166207,  606.36710479,  972.28374089, 1305.39426848,
    #    1173.89006446,  763.00684754, 1962.83663008, 2097.25095522,
    #    1605.16831696, 1766.47094034, 2614.32655741, 1978.74326398,
    #    2101.71978931, 3313.20579709, 3214.15074071, 3103.58374137,
    #    3481.27079111, 2794.18445722, 2909.00866364, 3224.77517999,
    #    3697.95111016, 3820.54821322, 3971.59513065, 3866.12386479,
    #    4103.99105997, 3720.63622549, 4412.09273829, 4603.98391633,
    #    4720.50703633, 5007.00837108, 5139.05079039, 4579.00363123,
    #    5215.96390284, 4716.94067352, 5980.        , 5684.9104407 ,
    #    5465.41318411, 5344.81386826, 5572.00364228, 5912.25548511,
    #    5741.50054043])

    # 0.3
    p = 27484124.02576489
    turbine_x = np.array([ 100.        , 1749.97059471, 2026.67739302, 2798.63336231,
       2586.24624638, 5926.27243056, 5966.42820801,  100.        ,
       1197.47344958, 2350.74632264, 2575.26394158, 4205.78245031,
       4985.76073006, 5975.83927298,  100.        ,  985.6119631 ,
       1802.09851689, 2998.920502  , 3624.73645289, 4989.08450415,
       5974.92636012,  455.83777277,  677.29452038, 2282.38752513,
       3195.14204278, 4030.14275819, 4881.46426452, 5974.64906873,
        733.74202485,  916.04818338, 1995.6480694 , 3358.1040887 ,
       3878.40550191, 4067.69318818, 5928.69204115,  100.        ,
        881.14293782, 2511.82761871, 2857.29578786, 3991.55189012,
       4720.17792855, 5883.19158755,  100.        , 1849.97550606,
       2555.55105435, 2869.95022778, 3289.61100672, 4374.87521974,
       5907.03980464])
    turbine_y = np.array([ 253.45555414,  100.05786795,  173.48676496,  100.03685583,   
        291.97394111,  117.44196871,  401.71200831,  774.98944129,
       1487.84253367,  455.17863769,  632.76277212, 1310.89159651,
       1152.57616257,  764.85779477, 1949.53036583, 2080.38257094,
       1851.02263936, 2249.25194903, 2608.40328357, 1962.86730449,
       2085.53008979, 3187.58923787, 2990.85266706, 3140.53915991,
       3609.95336596, 2767.92666628, 2891.58374091, 3217.71684309,
       3351.30499034, 3837.19312256, 3988.32687658, 3865.20458613,
       4159.32597197, 3744.48174176, 4412.55665032, 4627.38682263,
       4755.40495579, 5048.06570851, 4815.33517167, 4572.4473967 ,
       5162.52891194, 4712.84164895, 5903.74686589, 5719.97501274,
       5330.96932034, 5228.13182937, 5583.25180921, 5975.71242994,
       5876.74416572])

    # 0.25
    # turbine_x = np.array([ 307.81433932,  692.85906587,  680.56498234, 5052.04009782,
    #    4976.67053341, 5949.43482662, 5980.        ,  250.81890343,
    #     799.95573417,  633.28867401, 3158.20147498, 4602.60915402,
    #    4992.68694319, 5980.        ,  106.38413304,  749.89639798,
    #    2226.49427664, 3001.90169077, 3614.79143294, 5055.26004941,
    #    5969.91765918, 1052.18361533, 1163.04139725, 2258.44649871,
    #    3151.5221628 , 4019.08221926, 4847.84390639, 5963.40522288,
    #    1059.53183919, 1131.76574572, 2021.53686988, 3396.42135655,
    #    3758.51017822, 4023.48882033, 5921.38649755,  100.54248854,
    #     998.54473575, 2632.90425423, 3014.1589271 , 3977.21002183,
    #    4736.79219287, 5844.65243975,  100.        , 1497.30573707,
    #    2655.32504433, 2952.81228558, 3330.84146319, 4381.87647004,
    #    5904.46386799])
    # turbine_y = np.array([1194.44189061, 1394.22482097, 1488.61563497,  100.84557586,
    #    1395.86339665,  222.26260384,  362.68838299, 1272.37031034,
    #    1369.68320634, 1319.31163517, 2328.33420041, 1499.7188935 ,
    #    1142.27832124,  752.5897464 , 1460.36026477, 1553.83775596,
    #    2541.21321841, 2431.59009978, 2648.79110281, 2018.73828795,
    #    2125.88739426, 3247.09008164, 3327.15195971, 3124.58964789,
    #    3500.62862675, 2777.20647776, 2910.67052818, 3206.58784696,
    #    3393.07080307, 3639.63587548, 3972.42111691, 3886.64534435,
    #    4092.14634448, 3693.04006291, 4408.95147364, 4951.66735326,
    #    5410.49822402, 4975.03119993, 5230.99051836, 4613.23476868,
    #    5254.7258109 , 4737.15671418, 5980.        , 5780.07371449,
    #    5206.77617807, 5104.76622853, 5644.83861549, 5980.        ,
    #    5763.78463406])
    turbine_x = turbine_x - min(turbine_x)
    turbine_y = turbine_y - min(turbine_y)

    fi.reinitialize(layout=[turbine_x,turbine_y])
    fi.calculate_wake()
    farm_power = fi.get_farm_power()

    # spacing = calc_spacing(turbine_x, turbine_y)
    # print("minimum spacing: ", min(spacing)/rotor_diameter)

    # print("optimized power: ", np.sum(farm_power))

    # # # Using the FlorisInterface functions for generating plots, run FLORIS
    # # # and extract 2D planes of data.
    horizontal_plane_2d = fi.calculate_horizontal_plane(80.0, x_bounds=(np.min(flow_x),np.max(flow_x)),y_bounds=(np.min(flow_y),np.max(flow_y)),x_resolution=200, y_resolution=100)
    plt.figure(1)
    ax1 = plt.gca()
    visualize_cut_plane(horizontal_plane_2d, ax=ax1, title="probability threshold: 0.3", color_bar=True)
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    plot_poly(excluded_polygons, ax=ax1)
    plot_turbines(turbine_x, turbine_y, rotor_diameter/2.0, ax=ax1)
    plt.xlim(0,6000)
    plt.ylim(0,6000)
    plt.savefig("thresh0.3.pdf", transparent=True)
    plt.show()