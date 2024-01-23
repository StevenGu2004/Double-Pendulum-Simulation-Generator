"""
Examples

This script demonstrates the features of the simulation generator working with a few test objects.

This script requires 'numpy' to be installed within the Python environment you are 
running this script in for setting the initial conditions of some examples.
"""

# TODO add more test cases for varying parameters!

from numpy import pi
from time import perf_counter
from double_pendulum_simulation_generator import DoublePendulumAnimated


def main():
    """Entry point for code to be ran."""
    # Sample simulations
    # dp_unnamed = DoublePendulumAnimated()
    # dp_unnamed.generate_dps(animating_energy=True, update_title_elements=False)
    # dp1 = DoublePendulumAnimated(t_end=5, name="dp1")
    # dp1.generate_dps(animating_dp=True, update_title_elements=False, dark_bg=True)
    # dp2 = DoublePendulumAnimated(n=5, fps=60, t_end=25, name="dp2", trace_size=10)
    # dp2.generate_dps(animating_energy=True, scrolling=True, dark_bg=False)
    # dp3 = DoublePendulumAnimated(n=2, fps=45, t_end=1, name="dp3", trace_size=200)
    # dp3.generate_dps(animating_energy=True, scrolling=False, dark_bg=True)
    # dp4 = DoublePendulumAnimated(n=3, fps=200, t_end=3, name="dp4", trace_size=0, variation=pi/3, ic=[3*pi/4, -4, 0, 6])
    # dp4.generate_dps(animating_energy=True, scrolling=False, dark_bg=True)
    # dp5 = DoublePendulumAnimated(n=32, fps=50, t_end=5, name="dp5", trace_size=5, variation=pi/64, ic=[pi/2, 0, pi/2, 0])
    # dp5.generate_dps()
    # dp6 = DoublePendulumAnimated(n=3, name="dp6", trace_size=12, p=[2,3,5,7,11], ic=[pi/4, -2, pi/16, -2])
    # dp6.generate_dps(animating_energy=True, scrolling=True, dark_bg=True)
    # dp7 = DoublePendulumAnimated(name="dp7", p=[1,1,1,1,-9.8])
    # dp7.generate_dps(animating_energy=True, update_title_elements=False)

    # Testing exceptions
    # exception1 = DoublePendulumAnimated(n=0)
    # exception2 = DoublePendulumAnimated(fps=0)
    # exception3 = DoublePendulumAnimated(t_end=0)
    # exception4 = DoublePendulumAnimated(p=3.0)
    # exception5 = DoublePendulumAnimated(p=[1.0,2,3,4,'5'])
    # exception6 = DoublePendulumAnimated(name=[])
    # exception7 = DoublePendulumAnimated(trace_size=1.5)
    # exception8 = DoublePendulumAnimated(variation=[1,2.3])
    # exception9 = DoublePendulumAnimated(ic=[1.0,[1],3,4])
    print("Simulation Generated.")


if __name__ == "__main__":
    start = perf_counter()
    print("Running examples...")
    main()
    end = perf_counter()
    print(
        f"Time taken for examples to run: {round(end-start, 2)}")
