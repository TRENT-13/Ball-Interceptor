# Video
[https://youtu.be/28HVs5rGuMo](https://youtu.be/28HVs5rGuMo)


Sample output
```
Estimated parameters:
Mass: 1.007 kg
Gravity: 8.000 m/s²
Drag coefficient: 0.030698
Initial velocity: (4.92, -8.86) m/s

Trajectory Cost (MSE) for first 20 frames: 0.051057 m²

MSE Costs by Method:
RK4: 0.051057 m²
Euler: 0.050665 m²
Adams-Bashforth: 0.051057 m²
Trapezoidal: 0.051057 m²
```

# Ball-Interceptor
None of the enviromental variables are known(such as gravity, resistance, mass and velocity), input is the video of 10 frames long, after that we use newton's shooting method to find the variables and recover the curve using RK4 and ball motion ODE, next we use a newtons shooting method to find the curavture of the ball to hit another ball at the right place, at the right moment, at this time enviromental variables are known and only velocity needs to be computed for the interceptor

user input the shooting position and the intercept position, for in depth analysis of the algorithm and program see the Final_2 paper in finals_project folder
