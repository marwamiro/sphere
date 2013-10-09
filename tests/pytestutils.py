#!/usr/bin/env python

from sphere import *
import subprocess
import sys

def passed():
    return "\tPassed"

def failed():
    raise Exception("Failed")
    return "\tFailed"

def compare(first, second, string):
  if (first == second):
    print(string + passed())
  else:
    print(string + failed())
    return(1)

def compareFloats(first, second, string, criterion=1e-5):
    if abs(first-second) < criterion:
        print(string + passed())
    else :
        print(string + failed())
        print("First: " + str(first))
        print("Second: " + str(second))
        print("Difference: " + str(second-first))
        return(1)

def compareNumpyArrays(first, second, string):
    if ((first == second).all()):
        print(string + passed())
    else :
        print(string + failed())
        return(1)


def cleanup(spherebin):
    'Remove temporary files'
    subprocess.call("rm -f ../input/" + spherebin.sid + ".bin", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + ".*.bin", shell=True)
    print("")


