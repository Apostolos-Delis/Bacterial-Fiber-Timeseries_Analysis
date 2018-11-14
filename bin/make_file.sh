#!/usr/bin/env bash

if [ $# -ne 1 ];
then
    echo "USAGE: make_file.sh <FILE>"
    echo "Must have a FILE arguement"
    exit -1
fi

FILE_NAME="../src/$1"

create_py_file(){
    echo "#!/usr/bin/env python3" > $1
    echo "# coding: utf8" >> $1
    echo >> $1
    echo >> $1
    echo "import numpy as np" >> $1
    echo "import pandas as pd" >> $1
    echo "import matplotlib.pyplot as plt" >> $1
    echo "import os" >> $1
    echo "import sys" >> $1
    echo >> $1
    echo "if __name__ == \"__main__\":" >> $1
    echo "    pass" >> $1
    echo >> $1
}

if [ -f $FILE_NAME ];
then  
    echo "ERROR: $FILE_NAME already exists"
    exit -1
fi

touch "$FILE_NAME"

create_py_file $FILE_NAME
chmod +x $FILE_NAME
