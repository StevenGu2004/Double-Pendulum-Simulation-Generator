# Double Pendulum Simulation Generator

## Create simulations of double pendulum systems animated using Matplotlib in Python.

This project simulates the motion of a double pendulum with equations of motion obtained using Lagrangian Mechanics and animations created using Matplotlib, then saves it as an '.mp4' file. Optionally, the user can also generate an animation for the energies time series of the pendulum system. 

## Demo images of the simulations
### Double pendulum in motion
![dp_demo](https://github.com/StevenGu2004/Double-Pendulum-Simulation-Generator/assets/93726536/69414bab-6cb3-47db-bbaf-cde2f1f91c5e)
### Time series for the energies of multiple double pendulum systems
![dp_energies_demo](https://github.com/StevenGu2004/Double-Pendulum-Simulation-Generator/assets/93726536/7ef72f9c-e7c6-40fe-a540-cdd1f73f7345)

## Notes
- The user must have 'ffmpeg' installed before generating any animations, as it is required in order to save the results in an '.mp4' format
- The animations could take a very long time to run if a long animation duration is passed (e.g. 20 seconds) but lowering the fps when creating the class object would help


## Known Issues (Work in Progress)
- Energy graphs are unlabeled 
- Minimum potential energy (U min) doesn't correlate to the lowest potential energy on the graph
- Run time needs to be heavily optimized to handle larger animation durations
- Missing a GUI for easier handling of generations
