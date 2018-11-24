#!/usr/bin/env bash

errormsg(){
    >&2 echo -e "\e[31m$1\e[0m"
}

command_exists(){
    command -v $1 >/dev/null 2>&1 || { 
        errormsg "$1 needs to be installed. Aborting."; exit 1; 
    }
}

command_exists pip

pip install numpy 

pip install matplotlib

pip install tensorflow 

pip install seaborn

pip install pandas

pip install sklearn

pip install scipy
