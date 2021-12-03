import subprocess
import sys

REPEAT_COUNT = 10 # number of times each sketch will be run
FILES = [
    "minimal.sk",
    "less_minimal.sk",
    "less_less_minimal.sk",
    "less_less_less_minimal.sk",
    "less_less_less_less_minimal.sk",
]

def main(out_file):
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
    f = open(out_file, 'w')
    f.write('\n'.join(write_strings))
    f.close()

if __name__ == "__main__":
    main(sys.argv[1])