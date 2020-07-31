# define a helix

import numpy as np
import pdb
import scipy.optimize

length = (
    80  # length: can be determined from end-to-end distance, doesn't need to be parameterized.
)
npoints = 16  # number of beads
noise = 0.2  # noise
points = np.arange(0, 1, 1.0 / npoints)  # parameterized path from 0 to 1
radius = 2  # radius to generate data # radius
pitch = 20  # pitch of the helix
c = np.zeros([npoints, 3])  # coodinates

direction = "inx"  # also inx and iny.  Could do a more general rotation at some point.
if direction == "inx":
    xi = [2, 1, 0]
if direction == "iny":
    xi = [1, 2, 0]
if direction == "inz":
    xi = [0, 1, 2]

# start rotated to make sure it still works
c[:, xi[0]] = radius * np.cos(pitch * points)  # r*cos(t)
c[:, xi[1]] = radius * np.sin(pitch * points)  # r*sin(t)
c[:, xi[2]] = length * points  # c*t

# add some noise
c += noise * np.random.normal(size=[npoints, 3])

# we have defined helical points
# now calculate the molecular vectors u - just difference between them
u = np.diff(c, axis=0)
for ui in u:
    ui /= np.sqrt(np.dot(ui, ui))  # normalize then

# we have unit vectors (directors) now
# Let's calculate second order tensor Q from the vectors.
Q = np.zeros([3, 3])
for ui in u:
    Q += 1.5 * np.outer(ui, ui)
Q /= len(u)
Q -= 0.5 * np.eye(3)

print("Q in original frame:")
print(Q)
vals, vecs = np.linalg.eig(Q)

# we assume the helix axis director is eigenvector corresponding to the largest eivenvalue,
# but it might not be the first one?
# eigenvectors are already normalized, we don't need to normalize

# Compute the three p2's
p2 = np.mean(np.dot(u, vecs), axis=0)

dirindices = np.argsort(np.abs(p2))

h = vecs[:, dirindices[2]]
l = vecs[:, dirindices[1]]
m = vecs[:, dirindices[0]]

print("eigenvalues", vals)
print("p2_h: {:.4f}, h: [{:.8f},{:.8f},{:.8f}]".format(p2[dirindices[2]], h[0], h[1], h[2]))
print("p2_l: {:.4f}, l: [{:.8f},{:.8f},{:.8f}]".format(p2[dirindices[1]], l[0], l[1], l[2]))
print("p2_m: {:.4f}, m: [{:.8f},{:.8f},{:.8f}]".format(p2[dirindices[0]], m[0], m[1], m[2]))

# p2 seems to work in the largest h

# rotate the helix itself into the new coordinates
# in many cases, this seems to not be a a great rotation.  It seems to
# start tilting the helix a bit in many cases. Not sure why!!!!

# rearrange to put the helix axis along the z.
S = np.zeros([3, 3])
S = vecs

cp = np.dot(c, S)
cp1 = cp[:, dirindices]
cp = cp1

up = np.dot(u, S)
up1 = up[:, dirindices]
up = up1

up2 = np.diff(cp, axis=0)
for upi in up2:
    upi /= np.sqrt(np.dot(upi, upi))  # normalize

print(
    "Check the directors the same?:", np.max(up - up2, axis=0)
)  # this should be nearly zero, just a check.
# Now compute the _Ferriani_ FQ; the Q matrix in the local coordinate of the helix

# Q = (a.i)*(a.j) - 1/2 delta_ij,  where i,k are l,m,h, the local frame determined above.
# alternate way to compute the Q in the rotated frame. Just rotate the frame, and compute the same Q.

# Doesn't appear to change the z direction at all.
FQ = np.zeros([3, 3])
for upi in up:
    FQ += 1.5 * np.outer(upi, upi)
FQ /= len(up)
FQ -= 0.5 * np.eye(3)

# Compute the average helical direction
avecos = np.mean(up[:, 2])
avesin = np.sqrt(1 - avecos ** 2)
# we want to rotate by arccos(avecos) in the plane that keeps the x and y in the same ratio.
# rotate in the plane defined by (0,0,1) and (cos(phi),sin(phi),0), to align
# normal is (0,0,1) cross (cos(phi),sin(phi))

zaxis = np.array([0, 0, 1])
upr = np.zeros(np.shape(u))
for i in range(np.shape(upr)[0]):
    scal = np.sqrt(1 - up[i, 2] ** 2)
    # project out into x,y plane
    ax1 = np.array([up[i, 0] / scal, up[i, 1] / scal, 0])
    # normal from the plane
    nm = np.cross(zaxis, ax1)  # the normal to the plane
    v = up[i]  # the vector to rotate
    # R(theta)v = nm(nm.v) + cos(theta) (nm x v) x nm + sin(-theta)(nm x v)  # from wikipedia
    upr[i] = nm * np.dot(nm, v) + avecos * np.cross(np.cross(nm, v), nm) - avesin * np.cross(nm, v)

# this seems frequently too high. Not sure it's great.
print("helical alignment: ", np.mean(upr[:, 2]))

######### now do fitting ***************

cmid = 0.5 * (cp[0:-1, :] + cp[1:, :])

z = cmid[0:, 2] / length

print("Q in helical frame:")
print(FQ)

# the local directors should be (parametric derivative), there is a phase term c.
# set k = l/r
# dx/dt = -r*p*sin(p*t+c) -> -p/sqrt(p^2+k^2) sin(p*t+c)
# dy/dt = r*p*cos(p*t+c) -> p/sqrt(p^2+k^2) cos(p*t+c)
# dz/dt = l ->        k/sqrt(p^2+k^2)

# n_x = -sin(v)sin(p*t+c)  . . . need to fit the phase
# n_y = sin(v)cos(p*t+c)
# n_z = cos(v)
# where v = np.arccos(k/np.sqrt(p^2+k^2))
# d/dp sin(v) = d/dp [p/sqrt(p^2+k^2)] = k^2 / (p^2+k^2)^(3/2)  = 1/p cos(v)^2 sin(v)
# d/dp cos(v) = d/dp [k/sqrt(p^2+k^2)] = -pk / (p^2+k^2)^(3/2)  = -1/p sin(v)^2 cos(v)
# d/dk sin(v) = d/dk [p/sqrt(p^2+k^2)] = -pk / (p^2+k^2)^(3/2)  = -1/k cos(v)^2 sin(v)
# d/dk cos(v) = d/dk [k/sqrt(p^2+k^2)] = p^2 / (p^2+k^2)^(3/2)  = 1/k sin(v)^2 cos(v)

# minimize \sum_i (cos(v)-u_iz)^2 + (sin(v)cos(p*t+c)-u_iy)^2 + (-sin(v)sin(p*t+c)-u_ix)^2
#         \sum_i (cos(v)^2 - 2*cos(v)u_iz + sin(v)^2cos(p*t+c)^2 - 2sin(v)cos(p*t+c)u_iy^2 + sin(v)^2 sin(p*t+c)^2 + 2sin(v)sin(p*t+c)u_ix
#         \sum_i (1 - 2*cos(v)u_iz - 2sin(v)cos(p*t+c)u_iy^2 + 2sin(v)sin(p*t+c)u_ix^2
# maximize \sum_i (cos(v)u_iz + sin(v)cos(p*t+c)u_iy - sin(v)sin(p*t+c)u_ix
#
# Derivatives: c, p, and k
#         do/dc = \sum_i d/dc (cos(v)u_iz + sin(v)cos(p*t+c)u_iy - sin(v)sin(p*t+c)u_ix)
#               = \sum_i sin(v) [-sin(p*t+c)u_iy - cos(p*t+c)u_ix]  (assume sin(v)!=0)
#               = \sum_i [sin(p*t+c)u_iy + cos(p*t+c)u_ix]
#
#         do/dp = \sum_i d/dp(cos(v)) u_iz + d/dp(sin(v))cos(p*t+c)u_iy - sin(v)*p*sin(p*t+c)u_iy - d/dp(sin(v))sin(p*t+c)u_ix - sin(v)*p*cos(p*t+c)u_ix
#               = \sum_i -1/p sin(v)^2 cos(v) u_iz + 1/p cos(v)^2 sin(v) cos(p*t+c)u_iy - p sin(v)*sin(p*t+c)u_iy - 1/p cos(v)^2 sin(v)sin(p*t+c)u_ix - sin(v)*p*cos(p*t+c) u_ix
#               = sin(v)/p \sum_i [-sin(v)cos(v)u_iz + cos(v)^2 cos(p*t+c)u_iy - p^2 sin(p*t+c)u_iy - cos(v)^2 sin(p*t+c)u_ix - p^2 cos(p*t+c)u_ix
#               = sin(v)/p \sum_i [-sin(v)cos(v)u_iz + [cos(v)^2 cos(p*t+c) - p^2 sin(p*t+c)]u_iy + [cos(v)^2 sin(p*t+c) - p^2 cos(p*t+c)] u_ix
#
#         do/dk = \sum_i d/dk (cos(v)u_iz + sin(v)cos(p*t+c)u_iy - sin(v)sin(p*t+c)u_ix
#               = \sum_i 1/k sin(v)^2 cos(v) u_iz -1/k cos(v)^2 sin(v) [ cos(p*t+c)u_iy - sin(p*t+c)u_ix]
#               = 1/k sin(v)cos(v) \sum_i [sin(v) u_iz - cos(v)[cos(p*t+c)u_iy - sin(p*t+c)u_ix]


def obj_for_cpv(x, up, z):
    c = x[0]
    p = x[1]
    k = x[2]
    ux = up[:, 0]
    uy = up[:, 1]
    uz = up[:, 2]
    r = np.sqrt(p * p + k * k)
    sinv = p / r
    cosv = k / r
    sinpz = np.sin(p * z + c)
    cospz = np.cos(p * z + c)

    arg = p * z + c
    vec = cosv * uz + sinv * (cospz * uy - sinpz * ux)
    min = -2 * np.sum(vec)
    min += np.sum(ux ** 2 + uy ** 2 + uz ** 2)
    min += len(z)

    return min


def root_for_cpk(x, up, z):  # might have broken at some point? Not currently using.
    c = x[0]
    p = x[1]
    k = x[2]
    ux = up[:, 0]
    uy = up[:, 1]
    uz = up[:, 2]
    r = np.sqrt(p * p + k * k)
    sinv = p / r
    cosv = k / r
    arg = p * z + c
    sinpz = np.sin(arg)
    cospz = np.cos(arg)

    p1 = -2 * sinv * np.sum(uy * sinpz + uz * cospz)
    p2 = (
        2
        * sinv
        / p
        * np.sum(
            -sinv * cosv * uz
            + (cosv * cosv * cospz - p * p * sinpz) * uy
            + (cosv * cosv * sinpz - p * p * cospz) * ux
        )
    )
    p3 = -2 * sinv * cosv / k * np.sum(sinv * uz - cosv * (cospz * uy - sinpz * ux))

    return np.array([p1, p2, p3])


# fit the curve instead of the directors. Don't use for now.

# the local directors should be (parametric derivative), there is a phase term c.
# set k = r/l
# dx/dt = r/l*sin(p*t+c) -> r/l*sin(p*t+c)
# dy/dt = r/l*cos(p*t+c) -> r/l*cos(p*t+c)
# dz/dt = t ->           -> t

# minimize \sum_i (t-c_iz/l)^2 + (k*cos(p*t+c)-c_ix/l)^2 + (k*sin(p*t+c)-c_iy/l)^2
#         \sum_i t^2 + k^2*cos(p*t+c)^2 + k^2*sin(p*t+c) - 2t*c_iz/l - 2 c_ix/l cos(p*t+c) -2 c_iy/l sin(p*t+c) + (c_ix/l)^2 + (c_iy/l)^2 + (c_iz/l)^2
#         \sum_i t^2 + k^2 - 2*t*c_iz - 2*k*cos(p*t+c)c_ix - 2*k*sin(p*t+c)c_iy

# def obj_for_c_cpv(x,ch,z):
#    c = x[0]
#    p = x[1]
#    k = x[2]
#    cx = ch[:,0]
#    cy = ch[:,1]
#    cz = ch[:,2]
#    sinpz = np.sin(p*z+c)
#    cospz = np.cos(p*z+c)
#
#    arg = p*z+c
#    min = np.sum(z*z + k*k)
#    vec = z*cz + k*cospz + k*sinpz
#    min -= 2*np.sum(vec)
#    min += np.sum(cx**2 + cy**2 + cz**2)
#
#    return min
# currently not fitting to this
# results = scipy.optimize.minimize(obj_for_c_cpv,[0,20,0.2],args=(up,z),method='nelder-mead')

# our fitting of avecos should give us one constraint for avecos = k / sqrt(p^2+k^2)
#                                                         avesin = p / sqrt(p^2+k^2)
#
# we estimate the radius from projecting out cp in the z direction
trialr = np.mean(np.sqrt(cp[:, 0] ** 2 + cp[:, 1] ** 2))
trialk = length / trialr
# avecos^2 = k^2 / p^2+k^2
# avecos^2*(p^2+k^2) = k^2
# avecos^2 p^2 = k^2(1 - avecos^2)
# p^2 = k^2(1 - avecos^2)/(avecos^2)
trialp = trialk * np.sqrt((1 - avecos ** 2) / (avecos ** 2))
print("trial radius for fit: {:.4f}, original radius: {:.4f}".format(trialr, radius))
print("trial pitch for fit:  {:.4f}, original pitch:  {:4f}".format(trialp, pitch))

results = scipy.optimize.minimize(
    obj_for_cpv, [0, trialp, trialk], args=(up, z), method="nelder-mead"
)
phase = results.x[0]
newpitch = results.x[1]
k = results.x[2]
arg = newpitch * z + phase
newradius = length / k

print("pitch:", newpitch)
print("phase:", results.x[0])
print("length/radius:", results.x[1])
print("radius:", newradius)
print("objective function:", obj_for_cpv(results.x, up, z))
print(
    "objective function with original parameters:", obj_for_cpv([0, pitch, length / radius], up, z)
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

curves = [c, cp, u, up, upr]
labels = [
    "helix (unrotated)",
    "helix (rotated)",
    "directors (unrotated)",
    "directors (rotated)",
    "directors (rotated to helix)",
]
for i in range(len(curves)):
    fig = plt.figure(i)
    curve = curves[i]
    label = labels[i]
    ax = fig.gca(projection="3d")
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label=label)
    ax.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.zlabel('z') # not defined?
    plt.show()
