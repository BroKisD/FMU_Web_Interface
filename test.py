from fmpy import read_model_description
from fmpy.util import fmu_info
import struct

fmu = r".\Samples\TE142x---FMU-Samples-main\PneumaticCylinderModel2.fmu"

print("Python bitness:", 8 * struct.calcsize("P"))  # expect 64 on most Conda installs
print("\n--- FMU INFO ---")
print(fmu_info(fmu))  # prints FMI version, ME/CS support, platforms, model identifiers
print("\n--- Inspect ---")
print(read_model_description(fmu))
