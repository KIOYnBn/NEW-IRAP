import ray
import time
import os
import subprocess
now_path = os.getcwd()
blast_path = os.path.dirname(__file__)
database_path = os.path.join(blast_path, 'blastDB')
import sys
sys.path.append(blast_path)
from Visual import visual_longcommand
from Load import load_reload_folder


requests = sys.argv[1:]
file, database, number, ev, out = requests[0], requests[1], requests[2], requests[3], requests[4]
folder = load_reload_folder(file)


ray.init(num_cpus=10)
start = time.time()


@ray.remote 
def ray_blast(file, database, number, ev, out):
    save_path = os.path.join(out, os.path.split(file)[-1].split('.')[0] + '.pssm')
    if os.path.split(save_path)[-1] not in os.listdir(os.path.split(save_path)[0]):
        database_path = os.path.join(os.path.join(blast_path, 'blastDB'), database)
        command = visual_longcommand(file, database_path, number, ev, save_path)
        outcode = subprocess.Popen(command, shell=True)
        if outcode.wait() != 0:
            print('\r\tProblems', end='', flush=True)


results_id = []
for i in folder:
    results_id.append(ray_blast.remote(i, database, number, ev, out))


ray.get(results_id)
ray.shutdown()    
print("等待时间: {}s".format(time.time()-start))

if 'A' in os.listdir(out):
    os.remove(os.path.join(out, 'A'))
if 'A' in os.listdir(now_path):
    os.remove(os.path.join(now_path, 'A'))
