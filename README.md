# OVS-Tracker
IROS2025 Submit

2025.03.12

We are in the process of removing sensitive information from the code, and we will update it in about two weeks. The current parameter settings are:

maximum velocity vm = 2.5 m/s, ωm = 1.5 rad/s
maximum acceleration am = 1.5 m/s
αm = 1.0 rad/s
weights of penalty J λd = 100, λf = 0.01, λs = 10−6
weights of penalty L µv = 100, µf = 0.1, µs = 0.01


# Simulation Update

2025.04.01

We have updated the simulation part of the algorithm. The experiment can now be completed using some basic Python libraries. It includes centralized global conflict resolution and distributed trajectory optimization. The optimization formula is in a simplified version, and we will release the full C++ version later.

### Directory Structure
- `sim/map`: Contains the map files for agent simulation.
- `sim/photo/10agent`: Holds the output images from the simulation.
- `sim/src`: This is where the code resides.
- `sim/demo`: Stores the simulation configuration files.

The following GIF file demonstrates the running effect of the simulation (the arrows represent the agents).

![Simulation Demo](demo.gif)

### Running the Simulation

2025.04.01

1. **Install Dependencies**
   - Install the basic libraries using `pip` according to the `import` statements in the `demo` files. For example, if you see `import numpy` in the code, you can install it with `pip install numpy`. You need to install all the libraries mentioned in the `import` sections.
2. **Run the Simulation**
   - Navigate to the `sim` directory. You can use the `cd` command in the terminal, for example, if your project is in a directory named `project` and the `sim` directory is inside it, you can run `cd project/sim`.
   - Then, execute the following command to run the simulation:
   ```bash
   python3 demo.py
   ```
3. **View the Output**
   - After the simulation is completed, the output will be saved in the `sim/photo/10agent` directory. You can view the generated images there.

