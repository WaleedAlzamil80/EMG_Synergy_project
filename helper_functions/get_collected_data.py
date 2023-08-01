def GetEMGData(path):

  """
  Description:

    This function reads the EMG and skeleton data from a given path and returns them as numpy arrays.

    The EMG data consists of four components: s, x, y, and z, each with shape (n, 12), where n is the number of samples.

    The skeleton data consists of two components: NposSke and NrotSke, each with shape (3, 21, n), where n is the same as the EMG data.

    The NposSke contains the positional data of the skeleton joints in 3D space.

    The NrotSke contains the rotational data of the skeleton joints in Euler angles (roll, pitch, yaw) in radians.

  Arguments:

    @path: is the path for the mat file

  Return: (s, x, y, z, NposSke, NrotSke)

    where:

      s: the signal data

      x, y, z: the corrdiantes of the sensor

      NposSke: the corrdinates of each segment in the Skeleton

      NposSke: the rotaional data of each segment in the Skeleton
  """

  # Import the required libraries
  import numpy as np 
  import scipy 
  from sklearn.utils import resample 
  from scipy.spatial.transform import Rotation as R

  # Reading the data from the path
  fing = scipy.io.loadmat(path)["Dynamic_10"][0][0]

  # Accessing the Emgs data
  sigEmg = fing[7][0][0][9] # 48 -> (4, 12) -> (emg, x, y, z), high frequency

  s = sigEmg[0::4].T # n by 12 
  x = sigEmg[1::4].T # n by 12 
  y = sigEmg[2::4].T # n by 12 
  z = sigEmg[3::4].T # n by 12

  # Accessing the Skeleton data
  rotSke = fing[6][0][0][7] # 4 * 21 low frequency needs upsampling to the sigEmg 
  posSke = fing[6][0][0][6] # 3 * 21 low frequency needs upsampling to the sigEmg

  # Upsampling the skeleton data to match the Emgs data length
  # The new length for the skeleton data
  m = sigEmg.shape[1]

  # Upsample both arrays using sklearn.utils.resample along the third axis
  NrotSke = resample(rotSke.T, replace=True, n_samples=m).T 
  NposSke = resample(posSke.T, replace=True, n_samples=m).T

  # Convert the NrotSke from quaternions to Euler angles using scipy Rotation object
  # Create a Rotation object from the quaternions with shape (N x 4)
  rot = R.from_quat(NrotSke.reshape(4, -1).T)

  # Convert the Rotation object to Euler angles in XYZ order with shape (N x 3)
  euler = rot.as_euler("xyz")

  # Reshape the Euler angles array to match the original shape of NrotSke (3, 21, n)
  NrotSke = euler.T.reshape(3, -1).reshape(3, NrotSke.shape[1], -1)

  # Return the EMG and skeleton data as numpy arrays
  return (s, x, y, z, NposSke, NrotSke)
