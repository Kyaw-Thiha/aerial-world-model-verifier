"""Track K: offline airframe models of the required lean, fitted on mocap truth.

Nothing in this package is used at test time except the fitted parameters.  The model
maps path kinematics (velocity, acceleration, jerk, heading) to the unit thrust axis;
the truth counterpart is the airframe's actual body z axis from the mocap or Leica
attitude.  No depicted lean, camera, mount or edit enters a fit.
"""
