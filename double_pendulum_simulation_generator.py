"""
Double Pendulum Simulation Generator

This script can be used to generate a simulation of the motion of a double pendulum 
with equations of motion obtained using Lagrangian Mechanics and animations created 
using Matplotlib, then saves it as an '.mp4' file. 
Optionally, the user can also generate an animation for the energies time series 
of the pendulum system. The double pendulum simulations are stored as a class object, 
with all the customizable initial conditions and add-ons to be passed as 
optional parameters when initializing the class object.

This script requires that 'scipy', 'numpy', 'matplotlib', and 'itertools' 
to be installed within the Python environment you are running this script in.

Example
-------
    See: examples.ipynb

Notes
-----
    The user must have 'ffmpeg' installed before generating any animations, 
    as it is required in order to save the results in an '.mp4' format.

    The animations could take a very long time to run if a long animation duration 
    is passed (e.g. 20 seconds). Lowering the fps when creating the class object could help.
"""

from itertools import cycle
import numpy as np
from scipy.integrate import odeint
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
# Stops matplotlib from generating an animation inline or other backends
matplotlib.use("Agg")


class DoublePendulumAnimated:
    """Create an object with a simulated double pendulum animation."""

    def __init__(self,
                 n: int = 1,
                 fps: float = 100,
                 t_end: float = 10,
                 p: list[float] = [1.0, 1.0, 1.0, 1.0, 9.8],
                 name: str = "",
                 trace_size: int = 20,
                 variation: float = np.pi/32,
                 ic: list[float] = [np.pi/2, -2.0, np.pi/2, -4.0]):
        """
        Initialize the double pendulum simulation object.

        Optional Parameters
        -------------------
        n :
            Number of double pendulums simulated
        fps : 
            Frame per second for the video created
        t_end : 
            Length of animation
        p : 
            Physical parameters for the system [m1, m2, l1, l2, g]
            Note: 
                positive g is gravity pointed downwards; all units use
                kg for mass, meter for length, and second for time
        name : 
            Name of the '.mp4' file created
        trace_size : 
            Number of trailing dots behind the lower mass blob (visual aesthetic)
        variation : 
            Variating spacing between the n-simulated double pendulums
        ic : 
            Initial condition for the system [p1, w1, p2, w2] 
            Note: counter-clockwise is the positive direction; all units use
                  radian for angular displacement, and second for time
        """

        # Exceptions
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")

        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer.")

        if not (isinstance(t_end, float) or isinstance(t_end, int)) or t_end <= 0:
            raise ValueError("t_end must be a positive float or int.")

        if not isinstance(p, list) or len(p) != 5 or \
                not all((isinstance(val, float) or isinstance(val, int)) for val in p) or \
                not all(val > 0 for val in p[:-1]):
            raise ValueError("p must be a list of five floats or integers representing "
                             "[m1, m2, l1, l2, g], "
                             "with positive values for all string lengths and masses.")

        if not isinstance(name, str):
            raise ValueError("name must be a str.")

        if not isinstance(trace_size, int) or trace_size < 0:
            raise ValueError("trace_size must be a non-negative integer.")

        if not (isinstance(variation, float) or isinstance(variation, int)):
            raise ValueError("variation must be a float or int.")

        if not isinstance(ic, list) or len(ic) != 4 or \
                not all((isinstance(val, float) or isinstance(val, int)) for val in ic):
            raise ValueError(
                "ic must be a list of four floats or ints representing [p1, w1, p2, w2].")

        self.n = n
        self.fps = fps
        self.t_end = t_end
        self.p = p
        if name == "":
            self.name = "unnamed dp " + str(id(self))
        else:
            self.name = name
        if trace_size == 0:
            self.trace_size = 1
        else:
            self.trace_size = trace_size
        self.variation = variation
        self.ic = ic

    def generate_dps(self,
                     animating_dp: bool = True,
                     animating_energy: bool = False,
                     scrolling: bool = False,
                     update_title_elements: bool = False,
                     dark_bg: bool = False) -> None:
        """ 
        Generate n double pendulum simulations animated and saved as a '.mp4' file.

        Optional Parameter
        ------------------
        animate_dp : 
            Generate an an animation of the motion of the system (if true) 
        animate_energy : 
            Generate an animated time series of the energies of the system (if true) 
        scrolling : 
            Animates the energy graphs dynamically by only displaying a chunk 
            (five seconds interval) of the graph at any given time (if true)
        update_title_elements : 
            Update the angular physical quantities in the title (if true)
        dark_bg : 
            Set the background color of the animation to be black instead of white (if true)

        """

        def create_figure(dp: bool = False,
                          energy: bool = False) -> tuple[plt.Figure, plt.Axes] | None:
            """
            Create a figure and axes object for animating.

            Optional Parameter
            ------------------
            dp : 
                Figure for animating double pendulum animation (if true) 
            energy : 
                Figure for animating time series energy graph (if true) 

            Return
            ------
            Figure object for the desired animation, return None otherwise
            """
            if dp:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            elif energy:
                fig, ax = plt.subplots(1, 1)
            if dp or energy:
                return fig, ax
            return

        def get_image(path: str,
                      zoom: float = 1) -> OffsetImage:
            """
            Turn an image file path into an image usable for AnnotationBbox with a desired size.

            Parameter
            ---------
            path : 
                Path of image in file folder

            Optional Parameter
            ------------------
            zoom : 
                Controls size of image

            Return
            ------ 
            Image to be used for AnnotationBbox
            """
            return OffsetImage(plt.imread(path), zoom=zoom)

        def dp_rhs(v: np.ndarray,
                   t: float,    # Needed for odeint
                   p: np.ndarray) -> list:
            """
            Define the differential equations for the double pendulum system.

            Parameters
            ----------
            v : 
                Vector of the state variables 
                v = [p1,w1,p2,w2]
            t : 
                Time
            p : 
                Vector of the parameters:
                p = [m1,m2,l1,l2,g]

            Return
            ------
            f :  
                List of the dp_rhs equations for the 4 ODES
            """
            p1, w1, p2, w2 = v
            m1, m2, l1, l2, g = p

            # Create f = (p1'=,w1'=,p2'=,w2'=):

            w1_dot = (-(m1+m2)*g*np.sin(p1)-m2*l2*w2*w2*np.sin(p1-p2)
                      - m2*np.cos(p1-p2)*l1*w1*w1*np.sin(p1-p2)+m2*g*np.cos(p1-p2)*np.sin(p2))\
                / ((m1+m2)*l1-m2*l1*(np.cos(p1-p2)*np.cos(p1-p2)))
            w2_dot = (l1*w1*w1*2*np.sin(p1-p2)-g *
                      np.sin(p2)-l1*w1_dot*np.cos(p1-p2))/l2

            f = [w1, w1_dot, w2, w2_dot]
            return f

        def get_x1y1x2y2(p1: np.ndarray,
                         p2: np.ndarray,
                         l1: float,
                         l2: float) -> tuple:
            """
            Get the 2D cartesian positions of the pendulum masses 
            using the angles and lengths of strings. 

            Parameters
            ----------
            p1 : 
                Angle made with vertical of upper mass
            p2 : 
                Angle made with vertical of lower mass 
            l1 : 
                String attaching fixed origin and upper mass
            l2 : 
                String attaching upper mass and lower mass

            Returns
            -------
            x1 : 
                X coordinate of upper mass
            y1 : 
                Y coordinate of upper mass
            x2 : 
                X coordinate of lower mass
            y2 : 
                Y coordinate of lower mass
            (All coordinates relative to the coordinate system used by the matplotlib.animation class)
            """
            return (l1*np.sin(p1),
                    -l1*np.cos(p1),
                    l1*np.sin(p1)+l2*np.sin(p2),
                    -l1*np.cos(p1)-l2*np.cos(p2))

        def next_color() -> str:
            """
            Get the next color in color_list.

            Return
            ------
            Hexcode color next in color_list
            """
            return next(color_cycle)

        def segmentation(time_series: list[float]) -> np.ndarray:
            """
            Turn given list of time series into segments to be used for plotting with LineCollection.

            Parameter
            ---------
            data_list : 
                List of time series to be segmented

            Return
            ------
            Segmented data for the time series
            """
            return np.stack((np.c_[t_interval[:-1], t_interval[1:]], np.c_[time_series[:-1], time_series[1:]]), axis=2)

        def animate_dp(i: float, origin: list, trace_list: list, trace_size: int,
                       dp: plt.Line2D, dpl: plt.Line2D, dpt: PathCollection,
                       x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                       p1: np.ndarray, w1: np.ndarray, p2: np.ndarray, w2: np.ndarray) -> None:
            """
            Helper function for animating the double pendulums. Updates the values of all dp graphs.

            Parameters
            ---------
            i : 
                Current frame
            origin : 
                Origin of animation (0,0)
            trace_list : 
                Position queue of the trailing dots
            trace_size : 
                Size of trace_list
            dp : 
                Collection of the double pendulum mass blobs to be animated
            dpl : 
                Collection of the black lines of the double pendulum
            dpt : 
                Collection of the traces of points
            x1 : X 
                coordinate of upper mass
            y1 : Y 
                coordinate of upper mass
            x2 : 
                X coordinate of lower mass
            y2 : 
                Y coordinate of lower mass
            p1 : 
                Angle made with vertical of upper mass
            w1 : 
                Angular velocity of upper mass
            p2 : 
                Angle made with vertical of lower mass 
            w2 : 
                Angular velocity of lower mass
            (All coordinates relative to the coordinate system used by the matplotlib.animation class)
            """
            dp.set_data([x1[i], x2[i]], [y1[i], y2[i]])
            dpl.set_data([origin[0], x1[i], x2[i]], [origin[1], y1[i], y2[i]])
            if len(trace_list) < trace_size:
                trace_list.append((x2[i], y2[i]))
            else:
                trace_list.pop(0)
                trace_list.append((x2[i], y2[i]))
            # Animating the trace
            dpt.set_offsets(np.array(trace_list))
            alphas = np.linspace(0, 1, len(trace_list))
            dpt.set_alpha(alphas)
            if self.n == 1 and update_title_elements:
                t = i*t_proportion  # Current t
                dp_ax.set_title(f"Double Pendulum Simulation \n"
                                f"$l_1$ = {round(l1, 2)},  "
                                f"$l_2$ = {round(l2, 2)},  "
                                f"$m_1$ = {round(m1, 2)},  "
                                f"$m_2$ = {round(m2, 2)},  "
                                f"$g$ = {round(g, 2)} \n"
                                f"$\phi_1({round(t, 2)})$ = {round(p1[i], 2)},  "
                                f"$\omega_1({round(t, 2)})$ = {round(w1[i], 2)},  "
                                f"$\phi_2({round(t, 2)})$ = {round(p2[i], 2)},  "
                                f"$\omega_2({round(t, 2)})$ = {round(w2[i], 2)}",
                                color=contrast_color)

        def animate_energy(i: float,
                           ke_lc: LineCollection,
                           pe_lc: LineCollection,
                           ke: np.ndarray,
                           pe: np.ndarray) -> None:
            """
            Helper function for animating the energies graphs.

            Parameters
            ----------
            i : 
                Current frame
            ke_lc : 
                LineCollection of the kinetic energy time series
            pe_lc : 
                LineCollection of the potential energy time series
            ke : 
                Time series of kinetic energy
            pe : 
                Time Series of potential energy
            """
            t = i*t_proportion  # Current t
            max_e = max(ke[i], pe[i])
            if scrolling:
                # Scrolling
                if t >= initial_xlim:
                    start = t - initial_xlim
                    end = t
                    energy_ax.set_xlim(start, end + x_padding)
                    energy_ax.xaxis.set_ticks(np.linspace(start, end, 5))
                    ann_e.set_position((end + x_padding + 0.05, -1.3))
                    ann_u.set_position(
                        (end + x_padding + 0.05, lowest_potential-1.3))

                if max_e >= current_ylim_dict['current_ylim']:
                    current_ylim_dict['current_ylim'] = max_e
                    y_lim = max_e + y_padding
                    energy_ax.set_ylim(lowest_potential -
                                       y_padding, y_lim + y_padding)
                    energy_ax.yaxis.set_ticks(np.linspace(
                        lowest_potential + y_padding, y_lim + y_padding, 5))

                if (t >= initial_xlim) or (max_e >= current_ylim_dict['current_ylim']):
                    energy_ax.tick_params(colors=contrast_color)
                    energy_ax.figure.canvas.draw()

            # Hides graph outside of allowed visibility region
            ke_lc.set_alpha(1*(t_interval <= t))
            pe_lc.set_alpha(1*(t_interval <= t))

        def animate_all_dps(i: float) -> None:
            """
            Animate all pendulums. 

            Parameter
            ---------
            i : 
                Current frame
            """
            for j in range(self.n):
                dp, dpl, dpt = dp_list[j]
                x1, y1, x2, y2 = position_list[j]
                p1, w1, p2, w2 = sol_list[j]
                animate_dp(i, origin, trace_list_list[j], self.trace_size,
                           dp, dpl, dpt, x1, y1, x2, y2, p1, w1, p2, w2)

        def animate_all_energies(i: float) -> None:
            """
            Animate all energy graphs. 

            Parameter
            ---------
            i : 
                Current frame
            """
            for j in range(self.n):
                current_ke_lc = ke_lc[j]
                current_pe_lc = pe_lc[j]
                current_ke = ke_list[j]
                current_pe = pe_list[j]
                animate_energy(i, current_ke_lc, current_pe_lc,
                               current_ke, current_pe)

        def set_fig_ax_parameters(set_dp: bool = False, set_energy: bool = False) -> None:
            """ 
            Set up the figure and axis for the respective animation. This method should 
            only ben called after the initialization of the respective figure and axis.

            Parameters
            ----------
            set_dp : 
                Set up the figure and axis of the double pendulum motion animation
            set_energy : 
                Set up the figure and axis of the time series energy graph animation
            """
            if set_dp:
                # To ensure these values don't crowd the animation,
                # they will only show up if only one pendulum is being animated.
                if self.n == 1:
                    dp_ax.set_title(f"Double Pendulum Simulation \n"
                                    f"$l_1$ = {round(l1, 2)},  "
                                    f"$l_2$ = {round(l2, 2)},  "
                                    f"$m_1$ = {round(m1, 2)},  "
                                    f"$m_2$ = {round(m2, 2)},  "
                                    f"$g$ = {round(g, 2)} \n"
                                    f"$\phi_1(0)$ = {round(self.ic[0], 2)},  "
                                    f"$\omega_1(0)$ = {round(self.ic[1], 2)},  "
                                    f"$\phi_2(0)$ = {round(self.ic[2], 2)},  "
                                    f"$\omega_2(0)$ = {round(self.ic[3], 2)}",
                                    color=contrast_color)
                else:
                    dp_ax.set_title(f"Double Pendulum Simulation \n"
                                    f"$l_1$ = {round(l1, 2)},  "
                                    f"$l_2$ = {round(l2, 2)},  "
                                    f"$m_1$ = {round(m1, 2)},  "
                                    f"$m_2$ = {round(m2, 2)},  "
                                    f"$g$ = {round(g, 2)}",
                                    color=contrast_color)
                dp_ax.set_xlabel("x (m)", color=contrast_color)
                dp_ax.set_ylabel("y (m)", color=contrast_color)

            if set_energy:
                energy_ax.set_title(
                    "Time Series for the Kinetic and Potential Energies of the System", color=contrast_color)
                energy_ax.set_xlabel("time (s)", color=contrast_color)
                energy_ax.set_ylabel("energy (J)", color=contrast_color)

        # Initializing variables
        m1, m2, l1, l2, g = self.p[0], self.p[1], self.p[2], self.p[3], self.p[4]
        frames = int(self.fps * self.t_end + 1)  # Number of frames/points
        t_interval = np.linspace(0, self.t_end, frames)
        t_proportion = self.t_end/(frames-1)  # Used for animating energies

        # Image Path
        hinge_path = 'nail.png'

        # Dp origin position
        origin = [0, 0]

        # Error tolerance (for odeint)
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Energy graphs
        linewidth = 4

        # Color palet from: https://venngage.com/blog/pastel-color-palettes/ (no. 4)
        color_list = ["#809bce", "#95b8d1", "#b8e0d2", "#d6eadf", "#eac4d5"]
        color_cycle = cycle(color_list)
        contrast_color = 'black'

        dp_fig, dp_ax = create_figure(dp=True)
        set_fig_ax_parameters(set_dp=True)

        if animating_energy:
            energy_fig, energy_ax = create_figure(energy=True)
            set_fig_ax_parameters(set_energy=True)
            lowest_potential = -abs(g)*(m1*l1 + m2*(l1+l2))
            x_padding = self.t_end/20
            y_padding = 15
            if scrolling:
                # Scrolling energy graph
                initial_xlim = self.t_end if self.t_end < 5 else 5
                energy_ax.set_xlim(-x_padding, initial_xlim)
                energy_x_start, energy_x_end = energy_ax.get_xlim()
                energy_ax.xaxis.set_ticks(np.linspace(
                    energy_x_start + x_padding, energy_x_end, 5))

                initial_ylim = 50
                current_ylim_dict = {'current_ylim': initial_ylim}
                energy_ax.set_ylim(lowest_potential - y_padding, initial_ylim)
                energy_y_start, energy_y_end = energy_ax.get_ylim()
                energy_ax.yaxis.set_ticks(np.linspace(
                    energy_y_start + y_padding, energy_y_end, 5))

            # Containers for animating energies
            ke_lc = []
            pe_lc = []
            ke_list = []
            pe_list = []

        if dark_bg:
            # Dark background
            contrast_color = "white"
            if animating_dp:
                dp_ax.set_facecolor("k")
                dp_fig.set_facecolor("black")
                dp_ax.set_facecolor("black")
                dp_ax.tick_params(colors=contrast_color)
                set_fig_ax_parameters(set_dp=True)

            if animating_energy:
                energy_ax.set_facecolor("k")
                energy_fig.set_facecolor("black")
                energy_ax.set_facecolor("black")
                set_fig_ax_parameters(set_energy=True)

        if animating_energy:
            energy_ax.axhline(0, color=contrast_color, zorder=1)
            energy_ax.axhline(lowest_potential,
                              color=contrast_color, linestyle='--', zorder=1)
            energy_ax.tick_params(colors=contrast_color)
            energy_ax.grid(color=contrast_color,
                           linewidth=0.4, alpha=0.3, zorder=0)
            if scrolling:
                ann_e = energy_ax.annotate(
                    "E=0", (energy_x_end + 0.05, -1.3), annotation_clip=False, color=contrast_color)
                ann_u = energy_ax.annotate(
                    "U min", (energy_x_end + 0.05, lowest_potential-1.3), annotation_clip=False, color=contrast_color)
            else:
                ann_e = energy_ax.annotate(
                    "E=0", (self.t_end + x_padding + 0.05, -1.3), annotation_clip=False, color=contrast_color)
                ann_u = energy_ax.annotate("U min", (self.t_end + x_padding + 0.05,
                                           lowest_potential-1.3), annotation_clip=False, color=contrast_color)

        if animating_dp:
            width = l1 + l2 + 0.5   # Width of graph window
            dp_ax.set_xlim(-width, width)
            dp_ax.set_ylim(-width, width)

            # Drawing the nail at the origin
            ab = AnnotationBbox(
                get_image(hinge_path, zoom=0.1), origin, frameon=False)
            dp_ax.add_artist(ab)

            # Containers for animating double pendulum
            position_list = []
            dp_list = []
            trace_list_list = [[] for i in range(self.n)]
            sol_list = []

        for i in range(self.n):
            # Initial conditions
            current_ic = self.ic.copy()
            current_ic[0] -= i*self.variation
            current_ic[2] -= i*self.variation

            # 2D array containing p1, w1, p2, w2
            sol = odeint(dp_rhs, current_ic, t_interval,
                         args=(self.p,), atol=abserr, rtol=relerr)
            p1, w1, p2, w2 = sol.T[0], sol.T[1], sol.T[2], sol.T[3]

            # 2D cartesian positions of the pendulum masses
            x1, y1, x2, y2 = get_x1y1x2y2(p1, p2, l1, l2)

            current_color = next_color()

            if animating_dp:
                # Initializing the double pendulum object containers
                position_list.append((x1, y1, x2, y2))
                sol_list.append((p1, w1, p2, w2))
                dp, = dp_ax.plot([], [], ms=25, c=current_color,
                                 marker='o', linestyle='', zorder=4)
                dpl, = dp_ax.plot([], [], c=contrast_color,
                                  alpha=0.6, zorder=2)
                dpt = dp_ax.scatter([], [], c=next_color())
                dp_list.append((dp, dpl, dpt))

            if animating_energy:
                # Time series' values for the energies
                ke = 0.5*m1*l1*l1*w1*w1 + 0.5*m2 * \
                    (l1*l1*w1*w1 + l2*l2*w2*w2 + 2*l1*l2*np.cos(p1-p2))
                pe = -(m1+m2)*g*l1*np.cos(p1) - m2*g*l2*np.cos(p2)
                ke_list.append(ke)
                pe_list.append(pe)
                ke_lc.append(LineCollection(segmentation(
                    ke), linewidths=linewidth, color=current_color))
                pe_lc.append(LineCollection(segmentation(
                    pe), linewidths=linewidth, color=current_color))
                ke_lc[-1].set_capstyle("round")
                pe_lc[-1].set_capstyle("round")
                energy_ax.add_collection(ke_lc[-1])
                energy_ax.add_collection(pe_lc[-1])

        if animating_dp:
            dp_ani = FuncAnimation(dp_fig,
                                   animate_all_dps,
                                   frames=frames-1,
                                   interval=1,
                                   blit=False)   # Set to False as it fails when self.n > 1

            dp_ani.save(self.name + '.mp4',
                        writer='ffmpeg',  # Install ffmpeg prior to running code
                        fps=self.fps)

        if animating_energy:
            energy_ax.autoscale_view()
            energy_ani = FuncAnimation(energy_fig,
                                       animate_all_energies,
                                       frames=frames-1,
                                       interval=1,
                                       blit=False)  # Doesn't work with the method chosen

            energy_ani.save(self.name + '_energies' + '.mp4',
                            writer='ffmpeg',  # Install ffmpeg prior to running code
                            fps=self.fps)

        if animating_dp or animating_energy:
            plt.close()
