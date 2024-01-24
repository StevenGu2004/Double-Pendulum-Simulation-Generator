# Double Pendulum Simulation Generator

## Create simulations of double pendulum systems animated using Matplotlib in Python.

This project simulates the motion of a double pendulum with equations of motion obtained using Lagrangian Mechanics and animations created using Matplotlib, then saves it as a '.mp4' file. Optionally, the user can also generate an animation for the energy time series of the pendulum system. 

## Demo images of the simulations
### Double pendulum in motion
https://github.com/StevenGu2004/Double-Pendulum-Simulation-Generator/assets/93726536/b5e107b1-69b5-4a44-b32c-a71fb49f289e

### Time series for the energies of multiple double pendulum systems
https://github.com/StevenGu2004/Double-Pendulum-Simulation-Generator/assets/93726536/6711f387-7a7d-4ef4-8ede-ea3653d0c1bb

## Notes
- The user must have 'ffmpeg' installed before generating any animations, as it is required to save the results in a '.mp4' format
- The animations could take a very long time to run if a long animation duration (e.g. 20 seconds) is used but lowering the fps when creating the class object would help


## Known Issues (Work in Progress)
- Run time needs to be heavily optimized to handle larger animation durations
- Missing a GUI for easier handling of generations
