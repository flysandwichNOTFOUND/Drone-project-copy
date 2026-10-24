# Drone electrical hazard inspection

Minimal scaffold for wire and plug inspection in indoor and outdoor scenes.
PyBullet owns physics, OpenCV will identify inspection targets, and a DeepSeek
adapter will assess structured detection evidence.

```text
drone_sim/
  config.py              # Simulation, camera and API settings
  models.py              # Shared detection and assessment records
  main.py                # Scaffold entry point
  pipeline.py            # Inspection workflow interface
  simulation/
    world.py             # PyBullet connection and stepping
    drone.py             # Drone loading, state and motor interfaces
    scenes.py            # Indoor/outdoor scene interfaces
  vision/
    camera.py            # Frame capture interface
    detector.py          # OpenCV wire/plug detection interface
  analysis/
    deepseek.py          # Asynchronous assessment interface
```

The old A* planner, custom dynamics, PID tuning tools, terrain routing, backend
comparisons and generated outputs have been removed.

## Run the scaffold

Use the same Python 3.10 Conda environment as the original project. Both local
VS Code workspaces select `C:\Users\frank\conda-envs\drone-sim\python.exe`.
From the copy folder, run:

```powershell
& 'C:\Users\frank\conda-envs\drone-sim\python.exe' -m pip install -r requirements.txt
& 'C:\Users\frank\conda-envs\drone-sim\python.exe' -m drone_sim.main
```

Only the configuration records and PyBullet lifecycle have basic implementations.
Scenes, assets, drone controls, capture, detection, API calls and the real-time
loop remain explicit stubs for subsequent instructions. The entry point prints
the scaffold status and does not launch an inspection.

Future API code will read `DEEPSEEK_API_KEY` from the environment. Model selection
and supported inputs must be established before implementing requests; the
current interface accepts structured evidence and does not assume image support.
API scheduling, request limits and handling stale assessments are also pending.
