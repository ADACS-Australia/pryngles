##################################################################
#                                                                #
# .#####...#####...##..##..##..##...####...##......######...####..#
# .##..##..##..##...####...###.##..##......##......##......##.....#
# .#####...#####.....##....##.###..##.###..##......####.....####..#
# .##......##..##....##....##..##..##..##..##......##..........##.#
# .##......##..##....##....##..##...####...######..######...####..#
# ................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################

# --------------------------------------------------
# External required packages
# --------------------------------------------------

from pryngles import *

import math
from colorsys import hls_to_rgb

import matplotlib.pyplot as plt

# Plotting in 3d
import numpy as np
from celluloid import Camera # getting the camera
from matplotlib import animation
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# --------------------------------------------------
# Class Plot
# --------------------------------------------------
class Plot:
    """Plotting util class"""

    def _pathpatch_2d_to_3d(pathpatch, pivot=None, zDir=None):
        """
        Create a patch in 3d around pivot in the direction of zDir

        Source: https://stackoverflow.com/a/69785236
        """

        if zDir is None:
            zDir = [0, 0, 1]
        if pivot is None:
            pivot = [0, 0, 0]
        path = pathpatch.get_path()  # Get the path and the associated transform
        trans = pathpatch.get_patch_transform()
        path = trans.transform_path(path)  # Apply the transform

        pathpatch.__class__ = mplot3d.art3d.PathPatch3D  # Change the class
        pathpatch._path2d = path  # Copy the 2d path
        pathpatch._code3d = path.codes  # Copy the codes
        pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

        # Get the 2D vertices and add the third dimension
        verts3d = np.empty((path.vertices.shape[0], 3))
        verts3d[:, 0:2] = path.vertices
        verts3d[:, 2] = pivot[2]

        # Get rotation matriz
        norm = np.linalg.norm(zDir)
        zDir = zDir / norm
        if np.abs(zDir[2]) == 1:
            yDir = np.array([0, zDir[2], 0])
        else:
            yDir = (np.array([0, 0, 1]) - zDir[2] * zDir) / math.sqrt(1 - zDir[2] ** 2)
        rotMat = np.empty((3, 3))
        rotMat[:, 0] = np.cross(zDir, yDir)
        rotMat[:, 1] = yDir
        rotMat[:, 2] = -zDir
        R = Rotation.from_matrix(rotMat)

        # Displace
        pathpatch._segment3d = R.apply(verts3d - pivot) + pivot

        return pathpatch

    # places a 3D circle in axes with 3d projection.
    def circle3d(ax, center=(0, 0, 0), radius=1, zDir="z", **kwargs):
        """
        Add a circle in 3d
        """
        pc = Circle(center[0:2], radius, **kwargs)
        ax.add_patch(Plot._pathpatch_2d_to_3d(pc, center, zDir))

    def pryngles_mark(ax):
        """Add a water mark to a 2d or 3d plot.

        Parameters:

            ax: Class axes:
                Axe where the pryngles mark will be placed.
        """
        # Get the height of axe
        axh = ax.get_window_extent().transformed(ax.get_figure().dpi_scale_trans.inverted()).height
        fig_factor = axh / 4

        # Options of the water mark
        args = dict(
            rotation=270, ha="left", va="top", transform=ax.transAxes, color="pink", fontsize=8 * fig_factor, zorder=100
        )

        # Text of the water mark
        mark = f"Pryngles {version}"

        # Choose the according to the fact it is a 2d or 3d plot
        try:
            ax.add_collection3d
            plt_text = ax.text2D
        except Exception:
            plt_text = ax.text

        text = plt_text(1, 1, mark, **args)
        return text

    def rgb(hls, to_hex=False):
        """Convert from hue (0-360), level (0-1) and saturation (0-1) to RGB

        Parameters:

            hls: array(3):
                Array with values of color:
                    hls[0]: hue, 0-360, see https://pythonfordesigners.com/tutorials/hsl-color-wheel/
                    hls[1]: level, 0: black, 1: white
                    hls[2]: saturation, 0: gray, 1: full-color

        Return:

            rgb: array(3):
                Array with rgb values (R: red, G: green, B: blue)
        """
        rgb_color = hls_to_rgb(hls[0] / 360.0, hls[1], hls[2])
        if to_hex:
            hex_color = f"#{int(rgb_color[0] * 255):02x}{int(rgb_color[1] * 255):02x}{int(rgb_color[2] * 255):02x}"
            return hex_color
        return rgb_color

    def rgb_sample(H=0):
        """Create a color table for a given hue"""
        fig, ax = plt.subplots(figsize=(9, 9))
        dL = 0.1
        dS = 0.1
        for S in np.arange(0, 1 + dS, dS):
            for L in np.arange(0, 1 + dL, dL):
                c = Circle((L, S), dL / 2.5, color=Plot.rgb([H, L, S]))
                ax.add_patch(c)
                ax.text(L, S, f"S={S:.1g},L={L:.1g}", ha="center", va="center", fontsize=6, color="y")
        ax.axis("off")
        ax.axis("equal")
        plt.tight_layout()

    # --------------------------------------------------
    # Tested methods from module file legacy
    # --------------------------------------------------

    def draw_pryngles(iobs=1, dark=False):
        ############################################################
        # Simulation
        ############################################################
        no = 10000
        P = RingedPlanet(
            Nr=500,
            Np=500,
            Nb=0,
            Rint=1.2,
            Rext=2.0,
            i=45 * DEG,
            a=0.1,
            e=0.1,
            lambq=70 * DEG,
            physics=dict(AL=1, AS=1, taug=1),
            behavior=dict(shadows=0),
        )
        P.changeObserver([90 * DEG, iobs * DEG])  # LOGO
        lamb_initial = 0.0 * DEG
        lamb_final = 360 * DEG
        lambs = np.linspace(lamb_initial, lamb_final, no)
        Rps = []
        Rrs = []
        ts = []
        Ts = []

        for lamb in lambs:
            P.changeStellarPosition(lamb)
            ts += [P.t * P.CU.UT]
            P.updateOpticalFactors()
            P.updateDiffuseReflection()
            P.updateTransit()

            Rps += [P.Rip.sum()]
            Rrs += [P.Rir.sum()]
            Tp = P.Tip.sum()

            T = Tp + P.Tir.sum()
            Ts += [T]

        Ts = np.array(Ts)
        ts = np.array(ts)
        Rps = np.array(Rps)
        Rrs = np.array(Rrs)
        ts = ts / Const.days

        ############################################################
        # Plot
        ############################################################
        ppm = 1e6
        alpha = 1

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
        # This title variable is not being used
        # title = (
        #     f"$a={P.a:g}$ au, "
        #     f"$i={P.i * RAD:g}^\\circ$ ($i_0={P.io * RAD:.1f}^\\circ$), "
        #     f"$\\lambda_\\mathrm{{q}}={P.lambq * RAD:g}^\\circ$, "
        #     f"Obs ($\\lambda$,$\\beta$) : ({P.eobs_ecl[0] * RAD:g}$^\\circ$, ""
        #     f"{P.eobs_ecl[1] * RAD:g}$^\\circ$)"
        # )
        theta = np.linspace(0, 2 * np.pi, no)
        x = np.cos(theta)
        y = np.sin(theta)
        z1 = ppm * (Rrs + Rps - 1e-3 * Ts)
        z2 = ppm * (Rps + Rps - 1e-3 * Ts)

        if dark:
            cmap = cmr.bubblegum
            back = "k"
            fcolor = "pink"
        else:
            cmap = cmr.bubblegum_r
            back = "w"
            fcolor = "blue"

        p = ax.scatter(x, y, (z2), marker=".", c=z1 + z2, s=7, cmap=cmap, alpha=alpha, edgecolors=None)
        cb = plt.colorbar(p, orientation="horizontal", fraction=0.03, pad=-0.2)
        cb.set_label(r"Pryngles", color=fcolor, fontsize=40, fontname="Special Elite")

        cb.ax.tick_params(labelcolor=back)
        cbytick_obj = plt.getp(cb.ax, "yticklabels")  # Set y tick label color
        plt.setp(cbytick_obj, color=back)
        cb.ax.tick_params(which="minor", length=2, color=back)
        cb.ax.tick_params(which="major", length=5, color=back)
        cb.update_ticks()

        # THE (SPHERICAL) SUN
        ax.scatter(0, 0, 0, marker=".", s=1000, color="orange")
        for i in np.linspace(0, 1, 20):
            ax.scatter(0, 0, 0, marker=".", s=1000 * 5 * i, color="gold", alpha=1 - 0.9 * i)

        # AXIS SETUP
        fig.set_facecolor(back)
        ax.set_axis_off()
        ax.set_facecolor(back)

        ##CAMERA ORIENTATION
        ax.view_init(elev=-30, azim=25)
        fig.tight_layout()

    # --------------------------------------------------
    # Tested methods from module file plot
    # --------------------------------------------------

    def calc_flyby(normal=None, start=0, stop=360, num=10, lat=0):
        """Calculate a flyby coordinates

        Parameters:
            normal: array (3), default = [0,0,1]:
                Normal to flyby plane.

            start: float, default = 0:
                Start longitude.

            stop: float, default = 0:
                Stop longitude.

            num: int, default = 10:
                Number of points in flyby.

            lat: float, default = 0:
                Constant latitude of flyby.
        """

        # Range of longitudes and latitudes
        if normal is None:
            normal = [0, 0, 1]
        lonp = np.linspace(start, stop, num)
        latp = lat * np.ones_like(lonp)

        # Rotation matrices
        _, Muni2sys = Science.rotation_matrix(normal, 0)

        # Compute directions
        nvecs = np.zeros((num, 3))
        for i in range(num):
            rp = Science.direction(lonp[i], latp[i])
            nvecs[i] = spy.mxv(Muni2sys, rp)

        return nvecs

    def animate_rebound(
        sim, filename=None, tini=0, tend=None, nsnap=None, interval=100, axis=False, traces=False, **plot_args
    ):
        """Animate a rebound simulation."""
        default_plot_args = dict(marker="o", color="r")
        default_plot_args.update(plot_args)

        verbosity = Verbose.VERBOSITY
        Verbose.VERBOSITY = VERB_NONE

        fig, ax = plt.subplots()

        if not traces:
            camera = Camera(fig)

        # Get the period of the longest osculant orbit
        P = -1
        for p in sim.particles[1:]:
            P = p.P if p.P > P else P

        # Choose properly tend and nsnap
        tend = P if tend is None else tend
        nsnap = int(tend / (P / 100)) if nsnap is None else nsnap

        if traces:
            sim.move_to_com()
            for p in sim.particles:
                xyz = p.xyz
                ax.plot(xyz[0], xyz[1], marker="*", color="k", ms=10, zorder=1000)

        # Simulate
        for i, t in enumerate(tqdm(np.linspace(tini, tend, nsnap))):
            sim.integrate(t)
            sim.move_to_com()

            for p in sim.particles:
                xyz = p.xyz
                ax.plot(xyz[0], xyz[1], **default_plot_args)

            if not traces:
                ax.text(
                    0.5,
                    1,
                    f"t = {sigfig.round(t, 3)} (snap {i + 1}/{nsnap})",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                )

                camera.snap()

        if axis:
            ax.grid()
        else:
            ax.axis("off")
        ax.axis("equal")

        if not traces:
            anim = camera.animate(interval=interval)
            Verbose.VERBOSITY = verbosity

            if filename is not None:
                if "gif" in filename:
                    anim.save(filename)
                    return anim
                elif "mp4" in filename:
                    ffmpeg = animation.writers["ffmpeg"]
                    metadata = dict(title="Pryngles Spangler Animation", artist="Matplotlib", comment="Movie")
                    w = ffmpeg(fps=15, metadata=metadata)
                    anim.save(filename, w)
                    return anim
                else:
                    raise ValueError(f"Animation format '{filename}' not recognized")
            else:
                return anim

            return anim
