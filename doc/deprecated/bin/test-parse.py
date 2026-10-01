#!/usr/bin/env python
##################################################################
#                                                                #
#.#####...#####...##..##..##..##...####...##......######...####..#
#.##..##..##..##...####...###.##..##......##......##......##.....#
#.#####...#####.....##....##.###..##.###..##......####.....####..#
#.##......##..##....##....##..##..##..##..##......##..........##.#
#.##......##..##....##....##..##...####...######..######...####..#
#................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################
"""This file parses the python code of a module searching for "inline"
test code.

Inline test code is code inserted into the source Jupyter notebooks of
module files.  Inline test code starts with "if IN_JUPYTER_TEST" line
and ends with
"unittest.main(argv=['first-arg-is-ignored'],exit=False)"

Example:

  import sys
  IN_JUPYTER='ipykernel' in sys.modules

  if IN_JUPYTER:
    def test_body(self):
        self.assertEqual([1],[1],True)

    class Test(unittest.TestCase):pass    
    Test.test_body=test_body
    unittest.main(argv=['first-arg-is-ignored'],exit=False)
"""
from sys import argv
import os

if len(argv)<=1:
    raise AssertionError("You must provide a filename.")
elif not os.path.exists(argv[1]):
    raise ValueError(f"Filename provided '{argv[1]}' not found.")

filepath=argv[1]
filename=os.path.basename(filepath)
filetest=f"/tmp/test-{filename}"
filesrc=f"/tmp/{filename}"

if os.path.exists(filetest):
    os.system(f"rm {filetest}")
if os.path.exists(filesrc):
    os.system(f"rm {filesrc}")

#Create
os.system(f"cat src/header.py > {filetest}")

fp=open(filepath,"r")
fo=open(filesrc,"w")
ft=open(filetest,"a")

ft.write("""import unittest
from pryngles import *
class Test(unittest.TestCase):
""")

qtest=False
qfulltest=False
qcont=False
qfullcont=False
ntest=0
for line in fp:
    if "#@test:" in line:
        ntest+=1
        qtest=True

    if "#@fulltest" in line:
        qfulltest=True
        ntest=1

    if qtest:
        if ("def test_" in line):
            qcont=True
        if ("class Test" in line):
            qcont=False
        if qcont:
            ft.write("\t"+line)
    elif qfulltest:
        if ("class Test" in line) or ("@fulltest" in line) or ("@end:fulltest" in line):
            qfullcont=False
        else:
            qfullcont=True
        if qfullcont:
            ft.write(line)
    else:
        fo.write(line)

    if "@end:test" in line:
        qtest=False

if not qfulltest:
    ft.write("""\
if __name__=="__main__":
    unittest.main(argv=['first-arg-is-ignored'],exit=False)
    """)

fp.close()
fo.close()
ft.close()

print(ntest)
