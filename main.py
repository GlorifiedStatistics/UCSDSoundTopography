from TopographicalMap import *
from SignalProcessing import *

obj_name = "TopographicalLineMap"

def write_triangles(path, triangles, o_name=obj_name):
    with open(path, 'w') as f:
        f.write("solid %s\n" % o_name)

        for v1, v2, v3 in triangles:
            f.write("facet normal 0.0 0.0 0.0\n")
            f.write("\touter loop\n")
            f.write("\t\tvertex %f %f %f\n" % v1)
            f.write("\t\tvertex %f %f %f\n" % v2)
            f.write("\t\tvertex %f %f %f\n" % v3)
            f.write("\tendloop\n")
            f.write("endfacet\n")

        f.write("endsolid")

path = './librarywalk.wav'
l, w, h = 8, 0.5, 0
num_samples = 50
base_ranges = ((20, 60), (61, 256), (257, 2047), (2048, 16384))
fs = 44100
num_bins = 20

"""
points = np.array(
    [get_single_note(path, num_samples, 8, n, 4, overtones=True, undertones=True)[:, 1] for n in NOTES])

points[points > 0.16] = 0.16


#points = np.array([get_range(path, num_samples, l, b)[:, 1] for b in base_ranges])

t_map = MultiTopographicalLineMap((l, w, h))
t_map.add_points(points)
"""

"""
t_map = TopographicalLineMap((l, w, h))
points = get_range(path, num_samples, l, (256, 20000))
"""

t_map = MultiTopographicalLineMap((l, w, h))
points = get_fourier(path, num_samples, func='sum', num_bins=num_bins, fs=fs)

t_map.add_points(points)

tris = t_map.get_triangles(normalize_height=3.0)

write_triangles("C:/Users/Babynado/Desktop/test.stl", tris)
