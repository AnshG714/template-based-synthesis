import subprocess
import os

REPEAT_COUNT = 10 # number of times each sketch will be run
OUT_FILE = "rev_list_performance.txt" #write the outputs for each file.
FILES = [
    "minimal.sk",
    "less_minimal.sk",
    "less_less_minimal.sk",
    "less_less_less_minimal.sk",
    "less_less_less_less_minimal.sk",
]

write_strings=[]
for file in FILES:
    runtimes = []
    for i in range(REPEAT_COUNT):
        print(f"Executing sketch for file {file} for trial {i + 1}")
        result = subprocess.run(["sketch", file, "--bnd-unroll-amnt=4"], stdout=subprocess.PIPE).stdout.decode('utf-8')
        time = result.split(" ")[-1][:-1]
        runtimes.append(time)
    write_strings.append(f"{file} {runtimes}")

# write to file
f = open(OUT_FILE, 'w')
f.write('\n'.join(write_strings))
f.close()
