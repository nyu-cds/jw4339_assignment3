#!/bin/bash
# a test tool for assignment 3

# echo 'arg1='$1
# echo 'arg2='$2
case "$1" in
    ""|-h|--help) 
        echo "Usage: ./profiler_tools.sh [<-h> | <-d|-t> <the_scprit.py>]"
        echo "       -h     print this help message"
        echo "       -d <py_script>   run diff"
        echo "       -t <py_script>   run timer"
        ;;
	-d) echo "execute \$diff original_out <(python $2)"
        diff original_out <(python $2)
        ;;
	-t) echo "execute \$sh -c 'python $2 > /dev/null'" 
        time sh -c 'python $2 > /dev/null'
        ;;
esac