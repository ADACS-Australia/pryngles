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
"""This file parses all the package code and creates a final release
version of the package.

It uses decorators of the type "#@" and the JupDev convention (see
DEVELOPERS.md)
"""
from sys import argv
import os
import re
#Tabulation
TAB=" "*4
COMBAR="#"+"%"*50

#Check input
if len(argv)<=1:
    raise AssertionError("You must provide a filename.")
elif not os.path.exists(argv[1]):
    raise ValueError(f"Filename provided '{argv[1]}' not found.")

#Read and pasrse input
filepath=argv[1]
filename=os.path.basename(filepath)
module=filename.split(".")[0]
print(f"Parsing source file {filepath}")

#Open file
fp=open(filepath,"r")

#Structures to store information
floating=[]
externals=[]
standalone=[]
consts=dict()
docstrings=dict()
classes=dict()
methods=dict()

#Traverse file
qcapture=False
block=[]
for line in fp:

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #Detect decorators
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if False:
        pass
    elif "#@class" in line:
        qcapture=True
        qtype="class"
        print(f"Class found...")
        
    elif "#@external" in line:
        qcapture=True
        qtype="external"
        print("External code found...")
        
    elif "#@standalone" in line:
        qcapture=True
        qtype="standalone"
        print("Standalone code found...")
        
    elif "#@consts" in line:
        qcapture=True
        qtype="consts"
        modulename=line.strip("\n\r").split(":")[1]
        print("Constants code found...")
        
    elif "#@docstring" in line:
        qcapture=True
        qtype="docstring"
        classname=line.strip("\n\r").split(":")[1]
        print(f"Docstring found for class {classname}...")
        
    elif "#@method" in line:
        qcapture=True
        qtype="method"
        classname=line.strip("\n\r").split(":")[1]
        print(f"Method found of class {classname}...")
        
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #Close block of code
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    elif "#@end" in line:
        qcapture=False
        
        #Store block of code
        if False:
            pass

        elif qtype=="external":
            externals+=["\n"]+block

        elif qtype=="standalone":
            standalone+=["\n"]+block

        elif qtype=="consts":
            if modulename not in consts:
                consts[modulename]=[]
            else:
                consts[modulename]+=["\n"]
            consts[modulename]+=block
            
        elif qtype=="docstring":
            docstrings[classname]=block

        elif qtype=="class":
            pattern=re.compile('^class ([\w^\(]+)\(', re.DOTALL)
            classname=pattern.findall(block[0])[0]
            classes[classname]=block

        elif qtype=="method":
            if classname not in methods:
                methods[classname]=[]
            else:
                methods[classname]+=["\n"]
            methods[classname]+=block

        #Clean block
        block=[]

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #Capture content
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    else:
        if qcapture:
            #Lines to exclude
            if "# #" in line:
                print("Excluding title...")
            block+=[line]
        else:
            floating+=[line]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Store contents
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
basepath=f"src/pryngles/.build/{module}"

#Target files
floating_file=f"{basepath}.floating"
fm=open(floating_file,"w")
for line in floating:
    fm.write(f"{line}")
fm.close()

external_file=f"{basepath}.external"
print(f"Saving externals into {external_file}...")
fm=open(external_file,"w")
if len(externals)>0:
    fm.write(f"""\n{COMBAR}\n# External required packages\n{COMBAR}\n""")
    for line in externals:
        fm.write(f"{line}")
fm.close()

sa_file=f"{basepath}.standalone"
print(f"Saving standalone into {sa_file}...")
fm=open(sa_file,"w")
if len(standalone)>0:
    fm.write(f"""\n{COMBAR}\n# Stand alone code of the module\n{COMBAR}\n""")
    for line in standalone:
        fm.write(f"{line}")
fm.close()

for modulename,constcontent in consts.items():
    consts_file=f"{basepath}_{modulename}.consts"
    print(f"Saving constants for {modulename} into {consts_file}...")
    fm=open(consts_file,"w")
    fm.write(f"""\n{COMBAR}\n# Constants of module {modulename}\n{COMBAR}\n""")
    for line in constcontent:
        fm.write(f"{line}")
    fm.close()

classes_file=f"{basepath}.classes"
fm=open(classes_file,"w")
for classname,classcontent in classes.items():

    #Saving class
    fm.write(f"{classname}\n")

    class_file=f"{basepath}_{classname}.class"
    print(f"Saving class {classname} into {class_file}...")
    fc=open(class_file,"w")
    fc.write("\n")
    fc.write(f"""\n{COMBAR}\n# Class {classname}\n{COMBAR}\n""")

    #Class code
    qhead=False
    for line in classcontent:

        if re.match("^class",line):
            qhead=True
            fc.write(line)

        if qhead and "):" in line:
            if classname in docstrings:
                fc.write(f'{TAB}"""')
                i=0
                for line in docstrings[classname]:
                    if i==0:
                        tab=f""
                        line=line.strip(" ")
                    else:
                        tab=f"{TAB}"
                    if "\"\"\"" in line:
                        pattern=re.compile('(.*)(?:""")(.*)', re.DOTALL)
                        for find in pattern.findall(line)[0]:
                            if find and ("=" not in find) and (";" not in find):
                                fc.write(f"{tab}{find}")
                                i+=1
                    else:
                        fc.write(f"{tab}{line}")
                        i+=1
                fc.write(f'{TAB}"""\n')
                fc.write(f"""\n{TAB}{COMBAR}\n{TAB}# Bassic methods\n{TAB}{COMBAR}\n""")
            qhead=False
            continue
        else:
            fc.write(line)

    fc.close()

fm.close()
    
for classname,methodcontent in methods.items():
    methods_file=f"{basepath}_{classname}.methods"
    print(f"Saving methods of {classname} into {methods_file}...")
    fm=open(methods_file,"w")
    fm.write(f"""\n{TAB}{COMBAR}\n{TAB}# Tested methods from module file {module}\n{TAB}{COMBAR}\n\n""")
    for line in methodcontent:
        fm.write(f"{TAB}{line}")
    fm.close()

