import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

end_indexes = []
directory = "wind_data"
dir_num = 0
for file in os.scandir(directory):
    if file.is_file():
        print("Decoding " + file.name)
        with open(directory + "/" + file.name, "rb") as binary_file:
            header = binary_file.read(19)  # Read header containing ascii characters
            # timestamp = []
            # u_velocity = []
            # v_velocity = []
            # w_velocity = []
            # speed_of_sound = []
            # temperature = []
            velocity_uvw = []
            while True:
                entry = binary_file.read(33)  # Read 33 byte repeating sequence

                # timestamp.append(
                #     int.from_bytes(entry[0:6], byteorder="big")
                # )  # First 6 bytes are timestamp (big endian) (seconds since 1904 Jan 1 epoch)

                u_velocity = (
                    int.from_bytes(entry[6:9], byteorder="big") / 1000 - 100
                )  # Convert to m/s and offset
                # next 3 bytes are u velocity (big endian)
                # Python adds leading 0s to make 4 bytes (int32) but if you want to write this in another lang you may need to add them

                v_velocity = (
                    int.from_bytes(entry[9:12], byteorder="big") / 1000 - 100
                )  # Convert to m/s and offset
                # next 3 bytes are v velocity (big endian)

                w_velocity = (
                    int.from_bytes(entry[12:15], byteorder="big") / 1000 - 100
                )  # Convert to m/s and offset
                # next 3 bytes are w velocity (big endian)

                # speed_of_sound.append(
                #     int.from_bytes(entry[15:18], byteorder="big")
                #     / 1000  # Convert to m/s
                # )  # next 3 bytes are speed of sound (big endian)

                # temperature.append(
                #     int.from_bytes(entry[18:21], byteorder="big") / 1000
                #     - 40  # Convert to deg C and offset
                # )  # next 3 bytes are temperature (big endian)

                velocity_uvw.append(np.array([u_velocity, v_velocity, w_velocity]))

                if not entry:
                    break
            del velocity_uvw[-1]

        if dir_num == 0:
            full_data = np.array(velocity_uvw)
            print(full_data.shape)
        else:
            full_data = np.concatenate((full_data, np.array(velocity_uvw)), axis=0)
        end_indexes.append(
            full_data.shape[0] - 1
        )  # save the index of the last element of that dataset

        dir_num += 1
        print("Finished decoding " + file.name)
        print("Decoded " + str(dir_num) + " files")

np.save("concatenated_data/all_data.npy", full_data)
np.save("concatenated_data/end_indeces.npy", np.array(end_indexes))
