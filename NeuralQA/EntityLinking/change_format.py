import argparse

import FileUtil

parser = argparse.ArgumentParser()
parser.add_argument("--filename")
args = parser.parse_args()
filename = args.filename

context = FileUtil.readFile(filename)
output = []
for i, c in enumerate(context):
    if i % 3 == 0:
        output.append("test-{} %%%% {}".format(int(i / 3 + 1), context[i + 1]))
FileUtil.writeFile(output, filename + ".query")
print("All done!")
