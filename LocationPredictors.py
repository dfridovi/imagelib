"""
Location predictors for eye detection.
"""

import numpy as np
import time

class LocationPredictorLPF:
	""" Adaptive low pass filter to predict next eye locations."""

	def __init__(self, start):
		self._t = None
		self._dt = 1.0 / 30

		self._x = start[1]
		self._y = start[0]

		self._vx = 0
		self._vy = 0

	def predict(self):

		# get timestep
		if self._t is not None:
			self._dt = time.time() - self._t
		self._t = time.time()

		# assume linear trajectory
		#self._x += self._vx * self._dt
		#self._y += self._vy * self._dt

		return self.getPos()

	def update(self, loc, score):

		# set inertia (weight of current location/velocity)
		inertia = np.exp(-0.5 * np.abs(score))

		# update position and velocity
		next_x = inertia*self._x + (1-inertia)*loc[1]
		next_y = inertia*self._y + (1-inertia)*loc[0]

		self._vx = inertia*self._vx + (1-inertia)*(loc[1]-self._x)/self._dt
		self._vy = inertia*self._vy + (1-inertia)*(loc[0]-self._y)/self._dt

		self._x = next_x
		self._y = next_y

	def getPos(self):
		return (self._y, self._x)

class LocationPredictorKalman:
	""" Kalman filter to predict next eye locations."""

	def __init__(self, start, AX_SD=5.0, AY_SD=5.0):
		self.ax_var = AX_SD**2
		self.ay_var = AY_SD**2

		self.last_t = None
		self.xstate = np.matrix([[start[1]],
								 [0]], dtype=np.float)
		self.ystate = np.matrix([[start[0]],
								 [0]], dtype=np.float)

		# Kalman filter parameters
		self.F = None
		self.Qx = None
		self.Qy = None
		self.Px = np.matrix([[10**2, 0],
							 [0, 0]], dtype=np.float)
		self.Py = np.matrix([[10**2, 0],
							 [0, 0]], dtype=np.float)
		self.H = np.matrix([[1, 0]], dtype=np.float)


	def setF(self, dt):
		self.F = np.matrix([[1, dt],
						    [0, 1]], dtype=np.float)

	def setQ(self, dt):
		self.Qx = self.ax_var * np.matrix([[dt**4 * 0.25, dt**3 * 0.5],
						  	 			   [dt**3 * 0.5, dt**2]], dtype=np.float)
		self.Qy = self.ay_var * np.matrix([[dt**4 * 0.25, dt**3 * 0.5],
						  	 			   [dt**3 * 0.5, dt**2]], dtype=np.float)

	def predict(self):
		if self.last_t is not None:
			dt = time.time() - self.last_t
		else:
			dt = 0
		self.last_t = time.time()

		self.setF(dt)
		self.setQ(dt)

		self.xstate = self.F * self.xstate
		self.ystate = self.F * self.ystate
		self.Px = self.F * self.Px * self.F.T + self.Qx
		self.Py = self.F * self.Py * self.F.T + self.Qy

		return self.getPos()

	def update(self, loc, score):
		R = np.matrix([[np.abs(1.0 / score) * 0.1]], dtype=np.float)

		Yx = np.matrix([[loc[1]]], dtype=np.float) - self.H * self.xstate
		Yy = np.matrix([[loc[0]]], dtype=np.float) - self.H * self.ystate

		Sx = self.H * self.Px * self.H.T + R
		Sy = self.H * self.Py * self.H.T + R
		Kx = self.Px * self.H.T * np.linalg.inv(Sx)
		Ky = self.Py * self.H.T * np.linalg.inv(Sy)

		self.xstate += Kx * Yx
		self.ystate += Ky * Yy

		self.Px -= Kx * self.H * self.Px
		self.Py -= Ky * self.H * self.Py

	def getPos(self):
		return (self.ystate[0, 0], self.xstate[0, 0])