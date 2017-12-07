import argparse
import xml.etree.ElementTree as ET
import multiprocessing
import subprocess
from pytocl.main import main
from combined_driver import Final_Driver
import os
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument('--drivers', nargs='+', type=str)
parser.add_argument('--tracks', nargs='+', type=str)
parser.add_argument('--length', type=int)
parser.add_argument('--laps', type=int)
args = parser.parse_args()

drivers = args.drivers
tracks = args.tracks
length = args.length
laps = args.laps
file_in = "quickrace.xml"

our_driver = ["src_server 1"]

all_drivers = ["berniw", "berniw3", "damned", "inferno", "lliaw", "tita", "berniw2", "bt", "inferno2", "olethros",  "sparkle"]
best_drivers = ["lliaw", "inferno", "olethros", "tita"]
best_driver = ["lliaw"]

all_tracks = ["aalborg", "alpine-1", "alpine-2", "brondehach", "corkscrew", "e-track-1", "e-track-2", "e-track-3", "e-track-4", "e-track-6", "eroad", "forza", "g-track-1", "g-track-2", "g-track-3", "ole-road-1", "ruudskogen", "spring", "street-1", "wheel-1", "wheel-2"]

# if not tracks or tracks[0] == "all":
#     tracks = all_tracks
#
# if drivers[0] == "best":
#     drivers = best_driver
# elif drivers[0] == "our":
#     drivers = our_driver

def set_drivers(root, drivers, port_offset):
    xml_drivers = root[4] #Drivers section
    del xml_drivers[3] #a specific driver definition
    for index, driver in enumerate(drivers):
        ele = ET.Element("section", attrib={"name": str(index + 1)})
        ele.append(ET.Element("attnum", attrib={"name": "idx", "val": str(port_offset)}))
        ele.append(ET.Element("attstr", attrib={"name": "module", "val": driver}))

        xml_drivers.append(ele)

def set_track(root, track):
    xml_track = root[1]
    xml_track[1][0].attrib["val"] = track

def write_and_run(tree, track, steering_values, max_speed, timeout, port):
    file_out = "tmp.quickrace.xml"
    # Write back to file
    tree.write(file_out)
    # Run TORCS

    # translate results
    # fitness = -running time*100 + distance done
    # print out result fitness
    steering_values = ", ".join([str(x) for x in steering_values])
    command = "./run_evaluation.sh '{}' {} {} {}".format(steering_values, max_speed, timeout, port)
    #["./run_evaluation.sh", steering_values, str(max_speed), str(timeout)]
    completed_command = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE)
    time.sleep(2)
    pids = list(completed_command.stdout.decode("utf-8").strip().split())
    print(pids)
    try:
        os.kill(pid[0], signal.SIGINT)
    except:
        pass

    try:
        os.kill(pid[1], signal.SIGINT)
    except:
        pass
    return completed_command

def run_on_ea_tracks(driver, steering_values, max_speed, timeout):
    return run_on_tracks(driver, ["ole-road-1", "alpine-1", "forza"], steering_values, max_speed, timeout)

def run_on_all_tracks(driver, steering_values, max_speed, timeout):
    return run_on_tracks(driver, all_tracks, steering_values, max_speed, timeout)

def run_on_tracks(driver, tracks, steering_values, max_speed, timeout):
    # Open original file
    tree = ET.parse(file_in)
    root = tree.getroot()

    client = []
    server = []
    server_temp_file = "server.out"
    client_temp_file = "client.out"
    base_port = 3001
    port_offset = 0
    for track in tracks:
        port = base_port + port_offset
        set_drivers(root, [driver], port_offset)
        if os.path.isfile(server_temp_file):
            os.remove(server_temp_file)
        if os.path.isfile(client_temp_file):
            os.remove(client_temp_file)
        set_track(root, track)
        completed_command = write_and_run(tree, track, steering_values, max_speed, timeout, port)
        with open("server.out") as f:
            server.append(f.readlines())
        with open("client.out") as f:
            client.append(list(f.readlines()))

        # we cycle through the ports
        port_offset += 1
        if base_port + port_offset > 3009:
            port_offset = 0

    print(server)
    return client, server

def get_distance_covered(client_out):
    distance = -1
    for index in range(len(client_out) - 1, 0, -1):
        if "distance" in client_out[index]:
            distance = int(float(client_out[index].strip().split("=")[1]))
            break
    if distance == -1:
        print(client_out)
    return distance

def get_total_time_covered(client_out):
    time = -1
    for index in range(len(client_out) - 1, 0, -1):
        if "time" in client_out[index]:
            time = int(float(client_out[index].strip().split("=")[1]))
            break
    return time
