import argparse
import xml.etree.ElementTree as ET
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--drivers', nargs='+', type=str)
parser.add_argument('--tracks', nargs='+', type=str)
parser.add_argument('--length', type=int)
parser.add_argument('--laps', type=int)
parser.add_argument('--file', type=str)
args = parser.parse_args()

drivers = args.drivers
tracks = args.tracks
length = args.length
laps = args.laps
file_in = args.file
file_out = "quickrace_auto_gen.xml"

all_drivers = ["berniw", "berniw3", "damned", "inferno", "lliaw", "tita", "berniw2", "bt", "inferno2", "olethros",  "sparkle"]
best_drivers = ["lliaw", "inferno", "olethros", "tita"]
best_driver = ["lliaw"]

all_tracks = ["aalborg", "alpine-1", "alpine-2", "brondehach", "corkscrew", "e-track-1", "e-track-2", "e-track-3", "e-track-4", "e-track-6", "eroad", "forza", "g-track-1", "g-track-2", "g-track-3", "ole-road-1", "ruudskogen", "spring", "street-1", "wheel-1", "wheel-2"]

if tracks[0] == "all":
    tracks = all_tracks

if drivers[0] == "best":
    drivers = best_driver

def set_drivers(root, drivers):
    xml_drivers = root[4] #Drivers section
    del xml_drivers[3] #a specific driver definition
    for index, driver in enumerate(drivers):
        ele = ET.Element("section", attrib={"name": str(index + 1)})
        ele.append(ET.Element("attnum", attrib={"name": "idx", "val": str(index + 1)}))
        ele.append(ET.Element("attstr", attrib={"name": "module", "val": driver}))

        xml_drivers.append(ele)

def set_track(root, track):
    xml_track = root[1]
    xml_track[1][0].attrib["val"] = track

def write_and_run(file_out, tree, track):
    # Write back to file
    tree.write(file_out)
    # Run TORCS
    command = "torcs -r ~/ci/torcs-client/" + file_out
    completed_command = subprocess.run(command, shell=True, check=True)
    command = "mv /tmp/lliaw_data.csv ./lliaw_"+track+".data"
    completed_command = subprocess.run(command, shell=True, check=True)

# Open original file
tree = ET.parse(file_in)
root = tree.getroot()

set_drivers(root, drivers)
for track in tracks:
    set_track(root, track)
    write_and_run(file_out, tree, track)
